import numpy as np
from itertools import product

def get_gc_content(seq):
    return round(((seq.count('C') + seq.count('G')) / len(seq)), 3)

def get_nt_comp(seq):
    bases = ['A', 'G', 'C', 'U']
    nt_percent = []
    for base_i in bases:
        nt_percent.append(round((seq.count(base_i) / len(seq)), 3))
    return nt_percent

def get_di_nt(seq):
    bases = ['A', 'G', 'C', 'U']
    pmt = list(product(bases, repeat=2))
    di_nt_percent = []
    for pmt_i in pmt:
        di_nt = pmt_i[0] + pmt_i[1]
        di_nt_percent.append(round((seq.count(di_nt) / len(seq)), 3))
    return di_nt_percent

def get_tri_nt(seq):
    bases = ['A', 'G', 'C', 'U']
    pmt = list(product(bases, repeat=3))
    tri_nt_percent = []
    for pmt_i in pmt:
        tri_nt = pmt_i[0] + pmt_i[1] + pmt_i[2]
        tri_nt_percent.append(round((seq.count(tri_nt) / len(seq)), 3))
    return tri_nt_percent

def get_tetra_nt(seq):
    bases = ['A', 'G', 'C', 'U']
    pmt = list(product(bases, repeat=4))
    tetra_nt_percent = []
    for pmt_i in pmt:
        tetra_nt = pmt_i[0] + pmt_i[1] + pmt_i[2] + pmt_i[3]
        tetra_nt_percent.append(round((seq.count(tetra_nt) / len(seq)), 3))
    return tetra_nt_percent

def get_dna_nt_comp(seq):
    bases = ['A', 'C', 'G', 'T']
    nt_percent = []
    for base_i in bases:
        nt_percent.append(round((seq.count(base_i) / len(seq)), 3))
    return nt_percent

def get_dna_di_nt(seq):
    bases = ['A', 'G', 'C', 'T']
    pmt = list(product(bases, repeat=2))
    di_nt_percent = []
    for pmt_i in pmt:
        di_nt = pmt_i[0] + pmt_i[1]
        di_nt_percent.append(round((seq.count(di_nt) / len(seq)), 3))
    return di_nt_percent

def get_pos_spec_nt(seq):
    bases = ['A', 'G', 'C', 'U']
    pos_base_vect = []
    for i in range(0, len(seq)):
        for base_i in bases:
            if seq[i] == base_i:
                pos_base_vect.append(1.0)
            else:
                pos_base_vect.append(0.0)
    return pos_base_vect

def start_pos_tri_nt(seq):
    bases = ['A', 'G', 'U', 'C']
    pmt = list(product(bases, repeat=3))
    startpos_tri_nt_vect = []
    start_pos_tri_nt = ''.join(seq[0:3])
    for tri_nt in pmt:
        tri_nt = ''.join(tri_nt)
        if start_pos_tri_nt == tri_nt:
            startpos_tri_nt_vect.append(1)
        else:
            startpos_tri_nt_vect.append(0)
    return startpos_tri_nt_vect

def start_pos_di_nt(seq):
    bases = ['A', 'G', 'U', 'C']
    pmt = list(product(bases, repeat=2))
    startpos_di_nt_vect = []
    startpos_di_nt = ''.join(seq[0:2])
    for di_nt in pmt:
        di_nt = ''.join(di_nt)
        if startpos_di_nt == di_nt:
            startpos_di_nt_vect.append(1)
        else:
            startpos_di_nt_vect.append(0)
    return startpos_di_nt_vect

# For secondary structure and energy, provide dummy values if not available
def sec_struct_composition(sec_struct):
    # sec_struct is a list of 1 (unpaired) and 2 (paired), length 19
    unpaired_comp = sec_struct.count(1) / len(sec_struct)
    paired_comp = sec_struct.count(2) / len(sec_struct)
    return [unpaired_comp, paired_comp]

def pos_spec_sec_struct(sec_struct):
    # sec_struct is a list of 1 (unpaired) and 2 (paired), length 19
    sec_struct_vect = []
    for s in sec_struct:
        sec_struct_vect.append(1 if s == 1 else 0)  # unpaired
        sec_struct_vect.append(1 if s == 2 else 0)  # paired
    return sec_struct_vect

def score_seq_by_pssm(pssm, seq):
    # pssm: 4 x n matrix, seq is a string of length n
    import math
    nt_order = {'A': 0, 'G': 1, 'C': 2, 'U': 3}
    scores = []
    for i, nt in enumerate(seq):
        if nt in nt_order:
            scores.append(pssm[nt_order[nt], i])
        else:
            # Assign uniform probability for unknown base (e.g., 'N')
            scores.append(0.25)
    log_score = sum([-math.log2(i) for i in scores])
    return log_score

def create_pssm(train_seq):
    train_seq = [list(seq) for seq in train_seq]
    train_seq = np.array(train_seq)
    nr, nc = np.shape(train_seq)
    pseudocount = nr ** 0.5
    bases = ['A', 'G', 'C', 'U']
    pssm = []
    for c in range(0, nc):
        col_c = train_seq[:, c].tolist()
        f_A = round(((col_c.count('A') + pseudocount) / (nr + pseudocount)), 3)
        f_G = round(((col_c.count('G') + pseudocount) / (nr + pseudocount)), 3)
        f_C = round(((col_c.count('C') + pseudocount) / (nr + pseudocount)), 3)
        f_U = round(((col_c.count('U') + pseudocount) / (nr + pseudocount)), 3)
        pssm.append([f_A, f_G, f_C, f_U])
    pssm = np.array(pssm).T  # shape (4, seq_len)
    return pssm

def extract_sirna_features(seq, sec_struct=None, energy=None, pssm=None, activity=None):
    """
    seq: string, full siRNA, e.g. 21 nt (19 RNA + 2 DNA)
    sec_struct: list of 19 ints (1 for unpaired, 2 for paired)
    energy: float
    pssm: 4 x 19 numpy array, or None (will use dummy if None)
    activity: float, optional
    """
    seq_len = len(seq)
    rna_seq = seq[0:seq_len-2].upper()
    dna_seq = seq[seq_len-2:seq_len].upper()

    features = []
    # 1. GC content
    features.append(get_gc_content(rna_seq))
    # 2. Nucleotide composition
    features.extend(get_nt_comp(rna_seq))
    # 3. Dinucleotide content
    features.extend(get_di_nt(rna_seq))
    # 4. Trinucleotide content
    features.extend(get_tri_nt(rna_seq))
    # 5. Tetranucleotide content
    features.extend(get_tetra_nt(rna_seq))
    # 6. DNA overhang nucleotide composition
    features.extend(get_dna_nt_comp(dna_seq))
    # 7. DNA overhang dinucleotide composition
    features.extend(get_dna_di_nt(dna_seq))
    # 8. Position-specific base comp
    features.extend(get_pos_spec_nt(rna_seq))
    # 9. Start position dinucleotide
    features.extend(start_pos_di_nt(rna_seq))
    # 10. Start position trinucleotide
    features.extend(start_pos_tri_nt(rna_seq))
    # 11/12. Secondary structure composition (dummy if not given)
    if sec_struct is None:
        sec_struct = [1]*len(rna_seq)  # All unpaired
    features.extend(sec_struct_composition(sec_struct))
    # 13. Position-specific secondary structure
    features.extend(pos_spec_sec_struct(sec_struct))
    # 14. Energy (dummy if not given)
    if energy is None:
        energy = 0.0
    features.append(energy)
    # 15. PSSM score (dummy PSSM if not given)
    if pssm is None:
        pssm = np.ones((4, len(rna_seq))) * 0.25  # Uniform
    features.append(score_seq_by_pssm(pssm, rna_seq))
    # 16. Activity (dummy if not given)
    if activity is None:
        activity = 0.0
    features.append(round(activity, 5))
    # 17. Potency indicator
    potent = 1 if activity is not None and activity >= 0.7 else 0
    features.append(potent)

    return np.array(features, dtype=np.float32)

def batch_extract_sirna_features(seqs, sec_structs=None, energies=None, pssm=None, activities=None):
    results = []
    for i, seq in enumerate(seqs):
        ss = sec_structs[i] if sec_structs is not None else None
        en = energies[i] if energies is not None else None
        act = activities[i] if activities is not None else None
        results.append(extract_sirna_features(seq, ss, en, pssm, act))
    return np.stack(results)