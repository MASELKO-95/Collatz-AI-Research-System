"""
Unit tests for the Transformer model architecture.

Tests cover:
- Model initialization
- Forward pass with different input shapes
- Output dimensions
- Model components (embedding, positional encoding, heads)
"""
import pytest
import torch
import torch.nn as nn
from model import CollatzTransformer, PositionalEncoding


class TestPositionalEncoding:
    """Test the PositionalEncoding module."""

    def test_initialization(self):
        """Test that PositionalEncoding initializes correctly."""
        d_model = 128
        max_len = 500
        pe = PositionalEncoding(d_model, max_len)
        
        assert hasattr(pe, "pe")
        assert pe.pe.shape == (max_len, 1, d_model)

    def test_forward_pass(self, device):
        """Test forward pass adds positional encoding."""
        d_model = 64
        batch_size = 8
        seq_len = 20
        
        pe = PositionalEncoding(d_model).to(device)
        x = torch.randn(seq_len, batch_size, d_model).to(device)
        
        output = pe(x)
        
        assert output.shape == x.shape
        # Output should be different from input (encoding added)
        assert not torch.allclose(output, x)

    def test_deterministic(self, device):
        """Test that positional encoding is deterministic."""
        d_model = 64
        seq_len = 30
        
        pe = PositionalEncoding(d_model).to(device)
        x = torch.randn(seq_len, 1, d_model).to(device)
        
        output1 = pe(x)
        output2 = pe(x)
        
        torch.testing.assert_close(output1, output2)


class TestCollatzTransformer:
    """Test the CollatzTransformer model."""

    def test_initialization(self, mock_model_config):
        """Test that model initializes with correct parameters."""
        model = CollatzTransformer(**mock_model_config)
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, "embedding")
        assert hasattr(model, "pos_encoder")
        assert hasattr(model, "transformer_encoder")
        assert hasattr(model, "stopping_time_head")
        assert hasattr(model, "next_step_head")

    def test_embedding_layer(self, mock_model_config):
        """Test that embedding layer has correct dimensions."""
        model = CollatzTransformer(**mock_model_config)
        
        # Embedding should map 3 tokens (0, 1, 2) to d_model dimensions
        assert model.embedding.num_embeddings == 3
        assert model.embedding.embedding_dim == mock_model_config["d_model"]

    def test_forward_pass_shape(self, mock_model_config, device):
        """Test that forward pass produces correct output shapes."""
        model = CollatzTransformer(**mock_model_config).to(device)
        
        batch_size = 4
        seq_len = 20
        
        # Create sample input (parity vectors)
        src = torch.randint(0, 2, (batch_size, seq_len)).to(device)
        
        stopping_time_pred, next_step_logits = model(src)
        
        # Check output shapes
        assert stopping_time_pred.shape == (batch_size, 1)
        assert next_step_logits.shape == (batch_size, seq_len, 3)

    def test_forward_pass_with_padding_mask(self, mock_model_config, device):
        """Test forward pass with padding mask."""
        model = CollatzTransformer(**mock_model_config).to(device)
        
        batch_size = 4
        seq_len = 20
        
        src = torch.randint(0, 2, (batch_size, seq_len)).to(device)
        # Create padding mask (True = padded position)
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
        padding_mask[:, 15:] = True  # Mask last 5 positions
        
        stopping_time_pred, next_step_logits = model(src, padding_mask)
        
        assert stopping_time_pred.shape == (batch_size, 1)
        assert next_step_logits.shape == (batch_size, seq_len, 3)

    def test_different_batch_sizes(self, mock_model_config, device):
        """Test that model handles different batch sizes."""
        model = CollatzTransformer(**mock_model_config).to(device)
        seq_len = 15
        
        for batch_size in [1, 2, 8, 16]:
            src = torch.randint(0, 2, (batch_size, seq_len)).to(device)
            stopping_time_pred, next_step_logits = model(src)
            
            assert stopping_time_pred.shape == (batch_size, 1)
            assert next_step_logits.shape == (batch_size, seq_len, 3)

    def test_different_sequence_lengths(self, mock_model_config, device):
        """Test that model handles different sequence lengths."""
        model = CollatzTransformer(**mock_model_config).to(device)
        batch_size = 4
        
        for seq_len in [5, 10, 50, 100]:
            if seq_len <= mock_model_config["max_len"]:
                src = torch.randint(0, 2, (batch_size, seq_len)).to(device)
                stopping_time_pred, next_step_logits = model(src)
                
                assert stopping_time_pred.shape == (batch_size, 1)
                assert next_step_logits.shape == (batch_size, seq_len, 3)

    def test_output_types(self, mock_model_config, device):
        """Test that outputs are tensors with correct dtype."""
        model = CollatzTransformer(**mock_model_config).to(device)
        
        batch_size = 4
        seq_len = 10
        src = torch.randint(0, 2, (batch_size, seq_len)).to(device)
        
        stopping_time_pred, next_step_logits = model(src)
        
        assert isinstance(stopping_time_pred, torch.Tensor)
        assert isinstance(next_step_logits, torch.Tensor)
        assert stopping_time_pred.dtype == torch.float32
        assert next_step_logits.dtype == torch.float32

    def test_gradient_flow(self, mock_model_config, device):
        """Test that gradients flow through the model."""
        model = CollatzTransformer(**mock_model_config).to(device)
        
        batch_size = 4
        seq_len = 10
        src = torch.randint(0, 2, (batch_size, seq_len)).to(device)
        
        stopping_time_pred, next_step_logits = model(src)
        
        # Compute dummy loss
        loss = stopping_time_pred.sum() + next_step_logits.sum()
        loss.backward()
        
        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_eval_mode(self, mock_model_config, device):
        """Test that model can switch to eval mode."""
        model = CollatzTransformer(**mock_model_config).to(device)
        
        model.eval()
        
        batch_size = 4
        seq_len = 10
        src = torch.randint(0, 2, (batch_size, seq_len)).to(device)
        
        with torch.no_grad():
            stopping_time_pred, next_step_logits = model(src)
        
        assert stopping_time_pred.shape == (batch_size, 1)
        assert next_step_logits.shape == (batch_size, seq_len, 3)

    def test_model_parameters_count(self, mock_model_config):
        """Test that model has reasonable number of parameters."""
        model = CollatzTransformer(**mock_model_config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params == total_params  # All params should be trainable

    def test_reproducibility_with_seed(self, mock_model_config, device):
        """Test that model produces same output with same seed."""
        torch.manual_seed(42)
        model1 = CollatzTransformer(**mock_model_config).to(device)
        
        torch.manual_seed(42)
        model2 = CollatzTransformer(**mock_model_config).to(device)
        
        src = torch.randint(0, 2, (4, 10)).to(device)
        
        with torch.no_grad():
            out1_stop, out1_next = model1(src)
            out2_stop, out2_next = model2(src)
        
        torch.testing.assert_close(out1_stop, out2_stop)
        torch.testing.assert_close(out1_next, out2_next)
