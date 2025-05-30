import requests
import json
import numpy as np

# Sample data
siRNA_seq = "AUGGCUAGCUAGCUAGCUA"  # 19 nucleotides
mRNA_seq = "AUGGCUAGCUAGCUAGCUAAUGGCUAGCUAGCUAGCUAAUGGCUAGCUAGCUAGCUA"  # 57 nucleotides

# Create feature maps with correct shapes
# siRNA_FM_data should be shape [19, 5]
siRNA_FM_data = np.random.rand(19, 5).tolist()

# mRNA_FM_data should be shape [57, 5]
mRNA_FM_data = np.random.rand(57, 5).tolist()

# td_data should be shape [24]
td_data = np.random.rand(24).tolist()

# Prepare the payload
payload = {
    "siRNA_sequence": siRNA_seq,
    "mRNA_sequence": mRNA_seq,
    "siRNA_FM_data": siRNA_FM_data,
    "mRNA_FM_data": mRNA_FM_data,
    "td_data": td_data
}

# Print shapes for debugging
print(f"siRNA_sequence length: {len(siRNA_seq)}")
print(f"mRNA_sequence length: {len(mRNA_seq)}")
print(f"siRNA_FM_data shape: {len(siRNA_FM_data)} x {len(siRNA_FM_data[0])}")
print(f"mRNA_FM_data shape: {len(mRNA_FM_data)} x {len(mRNA_FM_data[0])}")
print(f"td_data length: {len(td_data)}")

# Send the request
try:
    response = requests.post("http://127.0.0.1:8000/predict/", json=payload)
    
    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        print("Prediction successful!")
        print(f"Probabilities: {result['probabilities']}")
        
        # If the model returns class probabilities, the efficacy score is typically the probability of class 1
        if 'probabilities' in result and len(result['probabilities']) >= 2:
            efficacy_score = result['probabilities'][1]
            print(f"Efficacy score (class 1 probability): {efficacy_score}")
    else:
        print(f"Error sending request: {response}")
        print(f"Response status code: {response.status_code}")
        print(f"Response body: {response.text}")
except Exception as e:
    print(f"Exception occurred: {e}")

