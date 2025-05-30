import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re
import itertools
import math
import os
from collections import OrderedDict
from typing import List # Import List from typing
from contextlib import asynccontextmanager # Import asynccontextmanager

# Assuming your model definition is in scripts/model.py
# Corrected import path based on previous error
try:
    from scripts.model import Oligo
except ImportError:
    # Fallback import if model.py is in the current directory (less likely based on structure)
    # This fallback might still cause issues if not run as a package,
    # but the primary fix is the line above.
    from .scripts.model import Oligo


# --- Helper functions from infer.py and loader.py ---
# These functions are needed for data preprocessing

DeltaG = {'AA': -0.93, 'UU': -0.93, 'AU': -1.10, 'UA': -1.33, 'CU': -2.08, 'AG': -2.08, 'CA': -2.11, 'UG': -2.11, 'GU': -2.24,  'AC': -2.24, 'GA': -2.35,  'UC': -2.35, 'CG': -2.36, 'GG': -3.26, 'CC': -3.26, 'GC': -3.42, 'init': 4.09, 'endAU': 0.45, 'sym': 0.43}
DeltaH = {'AA': -6.82, 'UU': -6.82, 'AU': -9.38, 'UA': -7.69, 'CU': -10.48, 'AG': -10.48, 'CA': -10.44, 'UG': -10.44, 'GU': -11.40,  'AC': -11.40, 'GA': -12.44,  'UC': -12.44, 'CG': -10.64, 'GG': -13.39, 'CC': -13.39, 'GC': -14.88, 'init': 3.61, 'endAU': 3.72, 'sym': 0}

def antiRNA(RNA):
    antiRNA = []
    for i in RNA:
        if i == 'A' or i == 'a':
            antiRNA.append('U')
        elif i == 'U' or i == 'u' or i == 'T' or i == 't':
            antiRNA.append('A')
        elif i == 'C' or i == 'c':
            antiRNA.append('G')
        elif i == 'G' or i == 'g':
            antiRNA.append('C')
        elif i == 'X' or i == 'x':
            antiRNA.append('X')
    return ''.join(antiRNA[::-1])

def Calculate_DGH(seq):
    DG_all = 0
    DG_all += DeltaG['init']
    DG_all += ((seq[0] + seq[len(seq)-1]).count('A') + (seq[0] + seq[len(seq)-1]).count('U')) * DeltaG['endAU']
    DG_all += DeltaG['sym'] if antiRNA(seq).replace('T','U') == seq else 0
    for i in range(len(seq) - 1):
        DG_all += DeltaG[seq[i] + seq[i+1]]
    DH_all = 0
    DH_all += DeltaH['init']
    DH_all += ((seq[0] + seq[len(seq)-1]).count('A') + (seq[0] + seq[len(seq)-1]).count('U')) * DeltaH['endAU']
    DH_all += DeltaH['sym'] if antiRNA(seq).replace('T','U') == seq else 0
    for i in range(len(seq) - 1):
        DH_all += DeltaH[seq[i] + seq[i+1]]
    return DG_all,DH_all

def Calculate_end_diff(siRNA):
    count = 0
    _5 = siRNA[:2] # 5'end
    _3 = siRNA[-2:] # 3' end
    if _5 in ['AC','AG','UC','UG']:
        count += 1
    elif _5 in ['GA','GU','CA','CU']:
        count -= 1
    if _3 in ['AC','AG','UC','UG']:
        count += 1
    elif _3 in ['GA','GU','CA','CU']:
        count -= 1
    # Ensure keys exist in DeltaG before accessing
    dg_5 = DeltaG.get(_5, 0) # Use .get with a default value
    dg_3 = DeltaG.get(_3, 0) # Use .get with a default value
    return float('{:.2f}'.format(dg_5 - dg_3 + count * 0.45))


def calculate_td(df):
    # Ensure the input DataFrame has 'siRNA' and 'mRNA' columns
    if 'siRNA' not in df.columns or 'mRNA' not in df.columns:
        raise ValueError("Input DataFrame must contain 'siRNA' and 'mRNA' columns.")

    # Initialize new columns with default values (e.g., 0 or None)
    new_cols = ['ends', 'DG_1', 'DH_1', 'U_1', 'G_1', 'DH_all', 'U_all', 'UU_1', 'G_all', 'GG_1', 'GC_1', 'GG_all', 'DG_2', 'UA_all', 'U_2', 'C_1', 'CC_all', 'DG_18', 'CC_1', 'GC_all', 'CG_1', 'DG_13', 'UU_all', 'A_19']
    for col in new_cols:
        df[col] = 0.0 # Initialize with float 0.0

    for i in range(df.shape[0]):
        siRNA_seq = df.iloc[i, df.columns.get_loc('siRNA')] # Get siRNA sequence by column name
        if len(siRNA_seq) != 19:
             # Handle cases where siRNA length is not 19, maybe skip or raise error
             continue # Skipping for now, adjust as needed

        df.loc[i, 'ends'] = Calculate_end_diff(siRNA_seq)
        # Safely access DeltaG and DeltaH using .get()
        df.loc[i, 'DG_1'] = DeltaG.get(siRNA_seq[0:2], 0.0)
        df.loc[i, 'DH_1'] = DeltaH.get(siRNA_seq[0:2], 0.0)
        df.loc[i, 'U_1'] = int(siRNA_seq[0] == 'U')
        df.loc[i, 'G_1'] = int(siRNA_seq[0] == 'G')
        df.loc[i, 'DH_all'] = Calculate_DGH(siRNA_seq)[1]
        df.loc[i, 'U_all'] = siRNA_seq.count('U') / 19
        df.loc[i, 'UU_1'] = int(siRNA_seq[0:2] == 'UU')
        df.loc[i, 'G_all'] = siRNA_seq.count('G') / 19
        df.loc[i, 'GG_1'] = int(siRNA_seq[0:2] == 'GG')
        df.loc[i, 'GC_1'] = int(siRNA_seq[0:2] == 'GC')
        df.loc[i, 'GG_all'] = [siRNA_seq[j]+siRNA_seq[j+1] for j in range(18)].count('GG') / 18
        df.loc[i, 'DG_2'] = DeltaG.get(siRNA_seq[1:3], 0.0)
        df.loc[i, 'UA_all'] = [siRNA_seq[j]+siRNA_seq[j+1] for j in range(18)].count('UA') / 18
        df.loc[i, 'U_2'] = int(siRNA_seq[1] == 'U')
        df.loc[i, 'C_1'] = int(siRNA_seq[0] == 'C')
        df.loc[i, 'CC_all'] = [siRNA_seq[j]+siRNA_seq[j+1] for j in range(18)].count('CC') / 18
        df.loc[i, 'DG_18'] = DeltaG.get(siRNA_seq[17:19], 0.0)
        df.loc[i, 'CC_1'] = int(siRNA_seq[0:2] == 'CC')
        df.loc[i, 'GC_all'] = [siRNA_seq[j]+siRNA_seq[j+1] for j in range(18)].count('GC') / 18
        df.loc[i, 'CG_1'] = int(siRNA_seq[0:2] == 'CG')
        df.loc[i, 'DG_13'] = DeltaG.get(siRNA_seq[12:14], 0.0)
        df.loc[i, 'UU_all'] = [siRNA_seq[j]+siRNA_seq[j+1] for j in range(18)].count('UU') / 18
        df.loc[i, 'A_19'] = int(siRNA_seq[18] == 'A')

    df['td'] = df[new_cols].values.tolist() # Convert the calculated features to a list
    return df[['siRNA','mRNA','td']]


# This function is a simplified version of data_process_loader_infer
# It takes a single siRNA and mRNA pair and returns the processed tensors
def process_single_input(siRNA_seq: str, mRNA_seq: str, siRNA_FM_data: List[List[float]], mRNA_FM_data: List[List[float]], td_data: List[float]):
    # Tokenization and numericalization
    vocab = {'A': 1, 'U': 2, 'C': 3, 'G': 4, 'X': 0}

    def sequence_to_numerical(sequence, max_len):
        numerical_seq = [vocab.get(base, 0) for base in sequence.upper()]
        if len(numerical_seq) < max_len:
            numerical_seq.extend([0] * (max_len - len(numerical_seq)))
        elif len(numerical_seq) > max_len:
            numerical_seq = numerical_seq[:max_len]
        return numerical_seq

    siRNA_max_len = 19
    mRNA_max_len = 19 + 19 + 19

    # Convert sequences to feature maps with shape [batch_size, 1, seq_len, features]
    siRNA_tensor = torch.zeros(1, 1, siRNA_max_len, 5, dtype=torch.float32)
    mRNA_tensor = torch.zeros(1, 1, mRNA_max_len, 5, dtype=torch.float32)
    
    # Process sequences
    siRNA_numerical = sequence_to_numerical(siRNA_seq, siRNA_max_len)
    mRNA_numerical = sequence_to_numerical(mRNA_seq, mRNA_max_len)
    
    # Place the numerical values in the first feature column
    for i, val in enumerate(siRNA_numerical):
        siRNA_tensor[0, 0, i, 0] = float(val)
    
    for i, val in enumerate(mRNA_numerical):
        mRNA_tensor[0, 0, i, 0] = float(val)

    # Process FM data with shape [batch_size, 1, seq_len, features]
    siRNA_FM_tensor = torch.tensor(siRNA_FM_data, dtype=torch.float32)  # [19, 5]
    mRNA_FM_tensor = torch.tensor(mRNA_FM_data, dtype=torch.float32)    # [57, 5]
    
    # Add batch and channel dimensions
    siRNA_FM_tensor = siRNA_FM_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 19, 5]
    mRNA_FM_tensor = mRNA_FM_tensor.unsqueeze(0).unsqueeze(0)    # [1, 1, 57, 5]
    
    # Process td data
    td_tensor = torch.tensor(td_data, dtype=torch.float32).unsqueeze(0)  # [1, 24]

    # Print shapes for debugging
    print(f"siRNA_tensor shape: {siRNA_tensor.shape}")
    print(f"mRNA_tensor shape: {mRNA_tensor.shape}")
    print(f"siRNA_FM_tensor shape: {siRNA_FM_tensor.shape}")
    print(f"mRNA_FM_tensor shape: {mRNA_FM_tensor.shape}")
    print(f"td_tensor shape: {td_tensor.shape}")

    return siRNA_tensor, mRNA_tensor, siRNA_FM_tensor, mRNA_FM_tensor, td_tensor


def process_single_input_alt(siRNA_seq: str, mRNA_seq: str, siRNA_FM_data: List[List[float]], mRNA_FM_data: List[List[float]], td_data: List[float]):
    # Tokenization and numericalization
    vocab = {'A': 1, 'U': 2, 'C': 3, 'G': 4, 'X': 0}

    def sequence_to_numerical(sequence, max_len):
        numerical_seq = [vocab.get(base, 0) for base in sequence.upper()]
        if len(numerical_seq) < max_len:
            numerical_seq.extend([0] * (max_len - len(numerical_seq)))
        elif len(numerical_seq) > max_len:
            numerical_seq = numerical_seq[:max_len]
        return numerical_seq

    siRNA_max_len = 19
    mRNA_max_len = 19 + 19 + 19

    # Process sequences
    siRNA_numerical = sequence_to_numerical(siRNA_seq, siRNA_max_len)
    mRNA_numerical = sequence_to_numerical(mRNA_seq, mRNA_max_len)

    # Convert sequences to feature maps with shape [batch_size, 1, seq_len, features]
    siRNA_tensor = torch.zeros(1, 1, siRNA_max_len, 5, dtype=torch.float32)
    mRNA_tensor = torch.zeros(1, 1, mRNA_max_len, 5, dtype=torch.float32)
    
    # Place the numerical values in the first feature column
    for i, val in enumerate(siRNA_numerical):
        siRNA_tensor[0, 0, i, 0] = float(val)
    
    for i, val in enumerate(mRNA_numerical):
        mRNA_tensor[0, 0, i, 0] = float(val)

    # Process FM data with shape [batch_size, 1, seq_len, features]
    siRNA_FM_tensor = torch.tensor(siRNA_FM_data, dtype=torch.float32)  # [19, 5]
    mRNA_FM_tensor = torch.tensor(mRNA_FM_data, dtype=torch.float32)    # [57, 5]
    
    # Add batch and channel dimensions
    siRNA_FM_tensor = siRNA_FM_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 19, 5]
    mRNA_FM_tensor = mRNA_FM_tensor.unsqueeze(0).unsqueeze(0)    # [1, 1, 57, 5]
    
    # Process td data
    td_tensor = torch.tensor(td_data, dtype=torch.float32).unsqueeze(0)  # [1, 24]

    # Print shapes for debugging
    print(f"siRNA_tensor shape: {siRNA_tensor.shape}")
    print(f"mRNA_tensor shape: {mRNA_tensor.shape}")
    print(f"siRNA_FM_tensor shape: {siRNA_FM_tensor.shape}")
    print(f"mRNA_FM_tensor shape: {mRNA_FM_tensor.shape}")
    print(f"td_tensor shape: {td_tensor.shape}")

    return siRNA_tensor, mRNA_tensor, siRNA_FM_tensor, mRNA_FM_tensor, td_tensor


# --- FastAPI Application ---

# Define the input data structure expected by the API
class InputData(BaseModel):
    siRNA_sequence: str
    mRNA_sequence: str
    siRNA_FM_data: List[List[float]] # Should be of shape (19, 5)
    mRNA_FM_data: List[List[float]] # Should be of shape (57, 5)
    td_data: List[float] # Should be of shape (24,)


# Placeholder for the loaded model
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model when the FastAPI application starts and clean up on shutdown."""
    global model
    MODEL_PATH = "./model/best_model.pth" # Adjust path if necessary
    try:
        # Instantiate your model with the correct parameters
        base_model = Oligo(vocab_size=26, embedding_dim=128, lstm_dim=32, n_head=8, n_layers=1, lm1=19, lm2=19).to(device)
        base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
        # Wrap with our custom model
        model = CustomOligo(base_model).to(device)
        model.eval() # Set the model to evaluation mode
        print("Model loaded successfully!")
        yield # The application is running
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        model = None # Ensure model is None if loading fails
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None # Ensure model is None if loading fails
        yield # Allow the app to start but predictions will fail
    finally:
        # Clean up resources if needed on shutdown
        print("API shutting down.")


# Initialize FastAPI app with the lifespan event handler
app = FastAPI(lifespan=lifespan)


@app.post("/predict/")
async def predict(input_data: InputData):
    try:
        # Extract data from the request
        siRNA_seq = input_data.siRNA_sequence
        mRNA_seq = input_data.mRNA_sequence
        siRNA_FM_data = input_data.siRNA_FM_data
        mRNA_FM_data = input_data.mRNA_FM_data
        td_data = input_data.td_data
        
        # Print input data for debugging
        print(f"siRNA_sequence length: {len(siRNA_seq)}")
        print(f"mRNA_sequence length: {len(mRNA_seq)}")
        print(f"siRNA_FM_data shape: {len(siRNA_FM_data)} x {len(siRNA_FM_data[0])}")
        print(f"mRNA_FM_data shape: {len(mRNA_FM_data)} x {len(mRNA_FM_data[0])}")
        print(f"td_data length: {len(td_data)}")
        
        # Process input data
        siRNA_tensor, mRNA_tensor, siRNA_FM_tensor, mRNA_FM_tensor, td_tensor = process_single_input(
            siRNA_seq, mRNA_seq, siRNA_FM_data, mRNA_FM_data, td_data
        )
        
        # Make prediction
        with torch.no_grad():
            predictions, _, _ = model(
                siRNA_tensor,
                mRNA_tensor,
                siRNA_FM_tensor,
                mRNA_FM_tensor,
                td_tensor
            )
        
        # Extract probabilities
        probabilities = predictions.cpu().numpy().tolist()[0]
        
        # Calculate efficacy score (probability of class 1, scaled by 1.341 as in infer.py)
        efficacy_score = probabilities[1] * 1.341
        
        # Return the prediction
        return {
            "probabilities": probabilities,
            "efficacy_score": efficacy_score
        }
    except Exception as e:
        print(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# To run the API, save the code as api.py and run from your terminal:
# Make sure you are in the root directory of your project
# uvicorn api:app --reload
