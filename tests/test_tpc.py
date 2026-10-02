import torch
import pytest
from pyhealth.models.tpc import TPC

# 1. Mock the PyHealth dataset to prevent BaseModel crashes
class MockDataset:
    def __init__(self):
        # Tokens for categorical embeddings
        self.tokens = {"static_demographics": ["token1", "token2", "token3"]}
        
        # Add the schemas to satisfy BaseModel's validation checks
        self.input_schema = {
            "static_demographics": "categorical",
            "vitals": "continuous"
        }
        
        # Output schema defines the target variable type
        self.output_schema = {
            "los": "continuous"
        }
        
    def get_all_tokens(self, key):
        return self.tokens.get(key, [])

# 2. Parametrize to test dynamic shapes
@pytest.mark.parametrize("batch_size, seq_len, num_vitals, num_static", [
    (2, 24, 12, 1),  # Standard 24-hour window
    (4, 48, 12, 1),  # 48-hour window, larger batch
])
def test_tpc_forward_and_decay(batch_size, seq_len, num_vitals, num_static):
    # Initialize the mock dataset
    mock_dataset = MockDataset()
    
    # Generate the synthetic continuous data (vitals)
    vitals = torch.randn(batch_size, seq_len, num_vitals)
    
    # Generate the missingness tensors required by the Rocheteau paper
    # Binary mask: 1 if measured, 0 if missing
    vitals_mask = torch.randint(0, 2, (batch_size, seq_len, num_vitals)).float() 
    # Time since last measurement (random float between 0 and 5 hours)
    vitals_time_delta = torch.rand(batch_size, seq_len, num_vitals) * 5.0
    
    # Generate mock categorical static data (integers representing token indices)
    static_demographics = torch.randint(0, 3, (batch_size, num_static))
    
    # Mock targets (Length of Stay in days)
    mock_labels = torch.rand(batch_size, 1) * 10.0
    
    # Initialize the model
    model = TPC(
        dataset=mock_dataset,
        feature_keys=["static_demographics", "vitals"],
        label_key="los",
        mode="regression"
    )
    
    # Run the forward pass with the new time tensors
    output = model(
        static_demographics=static_demographics,
        vitals=vitals,
        vitals_mask=vitals_mask,
        vitals_time_delta=vitals_time_delta,
        y_true=mock_labels
    )
    
    # Assertions
    assert "loss" in output, "Model must return a loss for backpropagation"
    assert "y_prob" in output, "Model must return y_prob"
    assert output["y_prob"].shape == (batch_size, 1), "Output shape mismatch"
    assert not torch.isnan(output["loss"]), "Loss evaluates to NaN (check your decay function for div by zero)"