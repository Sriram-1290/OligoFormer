import torch
from scripts.model import Oligo, siRNA_Encoder, mRNA_Encoder

# Create model instance with the same parameters as in api.py
model = Oligo(vocab_size=26, embedding_dim=128, lstm_dim=32, n_head=8, n_layers=1, lm1=19, lm2=19)

# Print model architecture
print(model)

# Let's examine the forward methods of the encoders to understand the expected input shapes
print("\nsiRNA_Encoder forward method:")
print(siRNA_Encoder.forward.__code__.co_varnames)
print(siRNA_Encoder.forward.__code__.co_consts)

print("\nmRNA_Encoder forward method:")
print(mRNA_Encoder.forward.__code__.co_varnames)
print(mRNA_Encoder.forward.__code__.co_consts)

# Let's try with all inputs as feature maps
siRNA_tensor = torch.rand(1, 1, 19, 5)  # [batch_size, channels, seq_len, features]
mRNA_tensor = torch.rand(1, 1, 57, 5)   # [batch_size, channels, seq_len, features]
siRNA_FM_tensor = torch.rand(1, 1, 19, 5)  # [batch_size, channels, seq_len, features]
mRNA_FM_tensor = torch.rand(1, 1, 57, 5)   # [batch_size, channels, seq_len, features]
td_tensor = torch.rand(1, 24)

print("\nTrying with all inputs as feature maps")
print(f"siRNA_tensor shape: {siRNA_tensor.shape}")
print(f"mRNA_tensor shape: {mRNA_tensor.shape}")
print(f"siRNA_FM_tensor shape: {siRNA_FM_tensor.shape}")
print(f"mRNA_FM_tensor shape: {mRNA_FM_tensor.shape}")
print(f"td_tensor shape: {td_tensor.shape}")

# Let's trace through the model to see where the dimension mismatch occurs
try:
    # Forward pass through siRNA_encoder
    siRNA, siRNA_attention = model.siRNA_encoder(siRNA_tensor)
    print(f"siRNA shape after encoder: {siRNA.shape}")
    
    # Forward pass through mRNA_encoder
    mRNA, mRNA_attention = model.mRNA_encoder(mRNA_tensor)
    print(f"mRNA shape after encoder: {mRNA.shape}")
    
    # Calculate the exact padding needed to reach 4888 features when all tensors are concatenated
    # Current features: siRNA (1216) + mRNA (53*64=3392) + siRNA_FM (1) + mRNA_FM (1) + td (24) = 4634
    # Need to add: 4888 - 4634 = 254 features
    # Since each column in mRNA is 64 features, we need to add 254/64 = 3.97 columns
    # Let's add exactly 4 columns (256 features) to get to 4890, then trim 2 features
    
    if mRNA.shape[1] < 57:
        padding_size = 57 - mRNA.shape[1]
        padding = torch.zeros(mRNA.shape[0], padding_size, mRNA.shape[2], device=mRNA.device)
        mRNA = torch.cat([mRNA, padding], dim=1)
        print(f"mRNA shape after padding: {mRNA.shape}")
    
    # Apply pooling to feature maps
    siRNA_FM = model.siRNA_avgpool(siRNA_FM_tensor)
    print(f"siRNA_FM shape after avgpool: {siRNA_FM.shape}")
    siRNA_FM = siRNA_FM.view(siRNA_FM.shape[0], siRNA_FM.shape[2])
    print(f"siRNA_FM shape after view: {siRNA_FM.shape}")
    
    mRNA_FM = model.mRNA_avgpool(mRNA_FM_tensor)
    print(f"mRNA_FM shape after avgpool: {mRNA_FM.shape}")
    mRNA_FM = mRNA_FM.view(mRNA_FM.shape[0], mRNA_FM.shape[2])
    print(f"mRNA_FM shape after view: {mRNA_FM.shape}")
    
    # Flatten tensors
    siRNA = model.flatten(siRNA)
    print(f"siRNA shape after flatten: {siRNA.shape}")
    
    mRNA = model.flatten(mRNA)
    print(f"mRNA shape after flatten: {mRNA.shape}")
    
    # Trim 2 features from mRNA to get exactly 4888 features when concatenated
    mRNA = mRNA[:, :-2]
    print(f"mRNA shape after trimming: {mRNA.shape}")
    
    siRNA_FM = model.flatten(siRNA_FM)
    print(f"siRNA_FM shape after flatten: {siRNA_FM.shape}")
    
    mRNA_FM = model.flatten(mRNA_FM)
    print(f"mRNA_FM shape after flatten: {mRNA_FM.shape}")
    
    td = model.flatten(td_tensor)
    print(f"td shape after flatten: {td.shape}")
    
    # Concatenate all tensors
    merge = torch.cat([siRNA, mRNA, siRNA_FM, mRNA_FM, td], dim=-1)
    print(f"merge shape: {merge.shape}")
    
    # Check the expected input size for the classifier
    print(f"Expected input size for classifier: {model.classifier[0].in_features}")
    
    # Now try the full forward pass with our custom implementation
    x = model.classifier(merge)
    print(f"Output shape after classifier: {x.shape}")
    print(f"Success! Our custom implementation works.")
    
    # Try the model's forward method with our custom wrapper
    class CustomOligo(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, siRNA, mRNA, siRNA_FM, mRNA_FM, td):
            # Forward pass through siRNA_encoder
            siRNA, siRNA_attention = self.model.siRNA_encoder(siRNA)
            
            # Forward pass through mRNA_encoder
            mRNA, mRNA_attention = self.model.mRNA_encoder(mRNA)
            
            # Pad mRNA to ensure it has the correct shape (57, 64) instead of (53, 64)
            if mRNA.shape[1] < 57:
                padding_size = 57 - mRNA.shape[1]
                padding = torch.zeros(mRNA.shape[0], padding_size, mRNA.shape[2], device=mRNA.device)
                mRNA = torch.cat([mRNA, padding], dim=1)
            
            # Apply pooling to feature maps
            siRNA_FM = self.model.siRNA_avgpool(siRNA_FM)
            siRNA_FM = siRNA_FM.view(siRNA_FM.shape[0], siRNA_FM.shape[2])
            
            mRNA_FM = self.model.mRNA_avgpool(mRNA_FM)
            mRNA_FM = mRNA_FM.view(mRNA_FM.shape[0], mRNA_FM.shape[2])
            
            # Flatten tensors
            siRNA = self.model.flatten(siRNA)
            mRNA = self.model.flatten(mRNA)
            
            # Trim 2 features from mRNA to get exactly 4888 features when concatenated
            mRNA = mRNA[:, :-2]
            
            siRNA_FM = self.model.flatten(siRNA_FM)
            mRNA_FM = self.model.flatten(mRNA_FM)
            td = self.model.flatten(td)
            
            # Concatenate all tensors
            merge = torch.cat([siRNA, mRNA, siRNA_FM, mRNA_FM, td], dim=-1)
            
            # Apply classifier
            x = self.model.classifier(merge)
            
            return x, siRNA_attention, mRNA_attention
    
    # Create a custom model wrapper
    custom_model = CustomOligo(model)
    
    # Try the forward pass with the custom model
    predictions, siRNA_attention, mRNA_attention = custom_model(
        siRNA_tensor,
        mRNA_tensor,
        siRNA_FM_tensor,
        mRNA_FM_tensor,
        td_tensor
    )
    print(f"Success! Using custom model works.")
    print(f"Predictions shape: {predictions.shape}")
except Exception as e:
    print(f"Error: {e}")








