import numpy as np
import pandas as pd
from itertools import product
import math

"""
Calculate percentage GC content
"""
def get_gc_content(seq):
    return round(((seq.count('C') + seq.count('G'))/len(seq)),3)

"""
Get different nucleotide composition
"""
def get_nt_comp(seq):
    bases = ['A','G','C','U']
    nt_percent = []
    feat_names = []
    for base_i in bases:
        nt_percent.append(round((seq.count(base_i)/len(seq)),3))
        feat_names.append(base_i)
    return feat_names,nt_percent 

"""
Calculate the different percentages of di nucleotides
"""
def get_di_nt(seq):
    bases = ['A','G','C','U']
    pmt = list(product(bases,repeat=2))
    di_nt_percent = []
    feat_names = []
    for pmt_i in pmt:
        di_nt = pmt_i[0]+pmt_i[1]
        di_nt_percent.append(round((seq.count(di_nt)/len(seq)),3))
        feat_names.append(di_nt)
    return feat_names,di_nt_percent

"""
Calculate the different percentages of tri nucleotides
"""    
def get_tri_nt(seq):
    #print ("seq =", seq)
    bases = ['A','G','C','U']
    pmt = list(product(bases,repeat=3))
    tri_nt_percent = []
    feat_names = []
    for pmt_i in pmt:
        tri_nt = pmt_i[0]+pmt_i[1]+pmt_i[2]
        tri_nt_percent.append(round((seq.count(tri_nt)/len(seq)),3))
        feat_names.append(tri_nt)
    return feat_names,tri_nt_percent

"""
Calculate the different percentages of tetra nucleotides
"""    
def get_tetra_nt(seq):
    bases = ['A','G','C','U']
    pmt = list(product(bases,repeat=4))
    tetra_nt_percent = []
    feat_names = []
    for pmt_i in pmt:
        tetra_nt = pmt_i[0]+pmt_i[1]+pmt_i[2]+pmt_i[3]
        tetra_nt_percent.append(round((seq.count(tetra_nt)/len(seq)),3))
        feat_names.append(tetra_nt)
    return feat_names,tetra_nt_percent

"""
Calculate the nucleotide composition of the dna overhangs
"""
def get_dna_nt_comp(seq):
    bases = ['A', 'C', 'G', 'T']
    nt_percent = []
    feat_names = []
    for base_i in bases:
        nt_percent.append(round((seq.count(base_i)/len(seq)),3))
        feat_names.append(base_i + '_DNA')
    return feat_names,nt_percent   

"""
Calculate the di nucleotide composition of the dna overhangs
"""
def get_dna_di_nt(seq):
    bases = ['A','G','C','T']
    pmt = list(product(bases,repeat=2))
    di_nt_percent = []
    feat_names = []
    for pmt_i in pmt:
        di_nt = pmt_i[0]+pmt_i[1]
        di_nt_percent.append(round((seq.count(di_nt)/len(seq)),3))
        feat_names.append(di_nt + '_DNA')
    return feat_names,di_nt_percent

"""
Get the position specific base composition
"""
def get_pos_spec_nt(seq):
    bases = ['A', 'G', 'C', 'U']
    pos_base_vect = []
    feat_names = []
    for i in range(0,len(seq)):
        pos = i+1
        for base_i in bases:
            feat_names.append(base_i+str(pos))
            if seq[i] == base_i:
                pos_base_vect.append(1.000)
            else:
                pos_base_vect.append(0.000)
    #print("feat names =", feat_names)            
    return feat_names, pos_base_vect

"""
Get the starting position tri-nucleotide composition
"""
def start_pos_tri_nt(seq):
    bases = ['A', 'G', 'U', 'C']
    pmt = list(product(bases, repeat=3))
    startpos_tri_nt_vect = []
    feat_names = []
    start_pos_tri_nt = ''.join(seq[0:3])
    for tri_nt in pmt:
        tri_nt = ''.join(tri_nt)
            #print (tri_nt)
        feat_names.append(tri_nt+str(1))
        if start_pos_tri_nt == tri_nt:
            startpos_tri_nt_vect.append(1)
        else:
            startpos_tri_nt_vect.append(0)
    return feat_names, startpos_tri_nt_vect

"""
Get the starting position di-nucleotide composition
"""
def start_pos_di_nt(seq):
    bases = ['A', 'G', 'U', 'C']
    pmt = list(product(bases, repeat=2))
    startpos_di_nt_vect = []
    feat_names = []
    startpos_di_nt = ''.join(seq[0:2])
    for di_nt in pmt:
        di_nt = ''.join(di_nt)
        feat_names.append(di_nt+str(1))
        if startpos_di_nt == di_nt:
            startpos_di_nt_vect.append(1)
        else:
            startpos_di_nt_vect.append(0)
    return feat_names, startpos_di_nt_vect        

"""
Get the secondary structure composition of the RNA sequence 
"""
def sec_struct_composition(sec_struct):
    unpaired_comp = sec_struct.count(1)/len(sec_struct)
    paired_comp = sec_struct.count(2)/len(sec_struct)
    feat_names = ['UNPAIRED', 'PAIRED']
    return feat_names, [unpaired_comp, paired_comp]

"""
Get the position specific secondary structures
"""
def pos_spec_sec_struct(sec_struct):
    sec_struct_vect = []
    feat_names = []
    for i in range(0, len(sec_struct)):
        s = sec_struct[i]
        feat1 = 'UP'+str(i+1) # unpaired nucleotide at a given position
        feat_names.append(feat1)
        if s == 1:
            sec_struct_vect.append(1)
        else:
            sec_struct_vect.append(0)
            
        feat2 = 'P'+str(i+1) # paired nucleotide at a given position
        feat_names.append(feat2)
        if s == 2:
            sec_struct_vect.append(1)
        else:
            sec_struct_vect.append(0)
    return feat_names, sec_struct_vect       

"""
Score a given sequence by the pssm. We will use this score as a feature.
"""
def score_seq_by_pssm(pssm, seq):
    nt_order = {'A':0, 'G':1, 'C':2, 'U':3}
    ind_all = list(range(0,len(seq)))
    scores = [pssm[nt_order[nt],i] for nt,i in zip(seq,ind_all)]
    log_score = sum([-math.log2(i) for i in scores])
    return log_score

"""
Parse the RNA fold output
"""
def parse_rnafoldoutput(outputfile):
    fh = open(outputfile, 'r')
    sec_struct_vect = []
    energy_vect = []
    for line in fh:
        if len(line) < 5:
            continue
        if line.startswith('>'): # skip the header part
            continue
        elif 'A' in line or 'U' in line or 'C' in line or 'G' in line: # skip the sequence part
            continue
        else:
            sec_struct = line[0:19]
            sec_struct = sec_struct.replace(' ', '')
            sec_struct = sec_struct.replace('.', '1')
            sec_struct = sec_struct.replace('(', '2')
            sec_struct = sec_struct.replace(')', '2')
            sec_struct = list(sec_struct)
            sec_struct = [int(i) for i in sec_struct]
            energy = line[21:28]
            energy = energy.replace(' ', '')
            energy = energy.replace('(', '')
            energy = energy.replace(')', '')
            energy = float(energy)
            sec_struct_vect.append(sec_struct)
            energy_vect.append(energy)
    return sec_struct_vect, energy_vect

"""
Parse the data and get the training and test sets
"""
def parsedata(all_data, datatype):
    if datatype == 'train':
        col_ind = 2
    else:
        col_ind = 10
    all_nas = np.array(all_data[0])
    all_seqs = np.array(all_data[1])
    all_act = np.array(all_data[14])
    checked_col = np.array(all_data[col_ind]) # The column having information about which row is training data and which is test
                                             # data (marked as 'X').
    
    checked_ind = np.where(checked_col == 'X')[0]
    nas_subset = [int(all_nas[i]) for i in checked_ind]
    seq_subset = [all_seqs[i] for i in checked_ind]
    act_subset = [all_act[i] for i in checked_ind]
    
    return nas_subset, seq_subset, act_subset

"""
Write the siRNA sequences of the training and test data into a single fasta
file.
"""
def write_seq_to_fasta(nas_list,seq_list,outputfile):
    fh = open(outputfile, 'w')
    for nas,seq in zip(nas_list,seq_list):
        fh.write(">" + str(nas) + "\n" + seq[0:19] + "\n")
    fh.close()    

"""
Generate a position specific frequency table from the training data
"""    
def create_pssm(train_seq):
    #print(train_seq)
    train_seq = [list(seq) for seq in train_seq]
    train_seq = np.array(train_seq)
    #print(train_seq)
    nr,nc = np.shape(train_seq)
    pseudocount = nr**0.5 # Introduce a pseudocount (sqrt(N)) to make sure that we do not end up with a score of 0
    bases = ['A', 'G', 'C', 'U']
    pssm = []
    for c in range(0, nc-2):
        col_c = train_seq[:,c].tolist()
        f_A = round(((col_c.count('A') + pseudocount)/(nr+pseudocount)), 3)
        f_G = round(((col_c.count('G') + pseudocount)/(nr+pseudocount)), 3)
        f_C = round(((col_c.count('C') + pseudocount)/(nr+pseudocount)), 3)
        f_U = round(((col_c.count('U') + pseudocount)/(nr+pseudocount)), 3)
        pssm.append([f_A, f_G, f_C, f_U])
    pssm = np.array(pssm)
    pssm = pssm.transpose() # Make each column correspond to the nucleotide position and each row have the frequencies
                            # for different nucleotides at that position.
    #print (pssm)
    return pssm

"""
Calculate the different features for the training and test datasets
"""
def get_features(seq_list, activity_list, sec_struct_list, energy_list, pssm, nas_list):
    all_feat_headers = []
    all_feat_table = []
    headerflag = 0
    for seq,act,sec_struct,energy,nas in zip(seq_list, activity_list, sec_struct_list, energy_list, nas_list):
        seq_i_features = []
        seq_len = len(seq)
        rna_seq = seq[0:seq_len-2].upper() # The RNA sequence assuming that there are two dna bases at the end
        dna_seq = seq[seq_len-2:seq_len].upper() # The DNA overhang sequence

        # Get the GC content
        headername = 'GC content'
        gc = get_gc_content(rna_seq)
        
        if headerflag == 0:
            all_feat_headers.append(headername)
        seq_i_features.append(gc)
        
        # Get the individual base composition
        feat_names,nt_percent = get_nt_comp(rna_seq)
        seq_i_features.extend(nt_percent)
        if headerflag == 0:
            all_feat_headers.extend(feat_names)
            
        # Get the dinucleotide content
        feat_names,di_nt_percent = get_di_nt(rna_seq)
        seq_i_features.extend(di_nt_percent)
        if headerflag == 0:
            all_feat_headers.extend(feat_names)
               
        # Get the trinucleotide content
        feat_names,tri_nt_percent = get_tri_nt(rna_seq)
        seq_i_features.extend(tri_nt_percent)
        if headerflag == 0:
            all_feat_headers.extend(feat_names)
        
        # Get the tetranucleotide content
        feat_names,tetra_nt_percent = get_tetra_nt(rna_seq)
        seq_i_features.extend(tetra_nt_percent)
        if headerflag == 0:
            all_feat_headers.extend(feat_names)
            
        # Get the DNA overhang nucleotide composition
        feat_names,dna_nt_percent = get_dna_nt_comp(dna_seq)
        seq_i_features.extend(dna_nt_percent)
        if headerflag == 0:
            all_feat_headers.extend(feat_names)
        
        # Get the DNA overhang di-nucleotide composition
        feat_names,dna_di_nt_percent = get_dna_di_nt(dna_seq)
        seq_i_features.extend(dna_di_nt_percent)
        if headerflag == 0:
            all_feat_headers.extend(feat_names)
            
        # Get the position specific nucleotide vector
        feat_names,pos_base_vect = get_pos_spec_nt(rna_seq)
        if headerflag == 0:
            all_feat_headers.extend(feat_names)
        seq_i_features.extend(pos_base_vect)
        
        # Get the start position's dinucleotide composition
        feat_names, start_pos_di_nt_vect = start_pos_di_nt(rna_seq)
        if headerflag == 0:
            all_feat_headers.extend(feat_names)
        seq_i_features.extend(start_pos_di_nt_vect)        
        
        # Get the start position's trinulcleotide composition
        feat_names,start_pos_tri_nt_vect = start_pos_tri_nt(rna_seq)
        if headerflag == 0:
            all_feat_headers.extend(feat_names)
        seq_i_features.extend(start_pos_tri_nt_vect)    
        
        # Get the secondary structure composition
        feat_names, sec_struct_comp = sec_struct_composition(sec_struct)
        if headerflag == 0:
            all_feat_headers.extend(feat_names)
        seq_i_features.extend(sec_struct_comp)
        
        # Get the position specific secondary structure
        feat_names, sec_struct_vect = pos_spec_sec_struct(sec_struct)
        if headerflag == 0:
            all_feat_headers.extend(feat_names)
        seq_i_features.extend(sec_struct_vect)    
        
        # Include the energy calculated by RNAFold
        if headerflag == 0:
            all_feat_headers.append('ENERGY')
        seq_i_features.append(energy)    
        
        # Score the sequence by the PSSM generated from the training data
        pssm_score = score_seq_by_pssm(pssm, rna_seq)
        if headerflag == 0:
            all_feat_headers.append('PSSM_SCORE')
        seq_i_features.append(pssm_score)
        
        # Include the activity in the last column
        seq_i_features.append(round(act,5))
        if headerflag == 0:
             all_feat_headers.append('ACTIVITY')
        
        # Include a boolean feature to describe the siRNA as potent (act >= 0.7) or non-potent (act < 0.7)
        if act < 0.7:
            seq_i_features.append(0)
        else:
            seq_i_features.append(1)
        if headerflag == 0:
            all_feat_headers.append('POTENTYN')
        
        all_feat_table.append(seq_i_features)    
        headerflag = 1
    return all_feat_table, all_feat_headers

    
def main():
    datafile =  'Heusken_dataset.csv'
    all_data = pd.read_csv(datafile, sep=',', header=None)
    train_feature_file = 'training_features_check.csv'
    test_feature_file = 'test_features_check.csv'
    
    # Parse the training and test data
    datatype = 'train'
    train_nas,train_seq,train_activity = parsedata(all_data, datatype)
    
    # Get the frequencies of nucleotides at each position in the training set as a Position Specific Scoring Matrix. 
    # We will use this to score both training and test sequences and using this score as a feature.
    pssm = create_pssm(train_seq)

    datatype = 'test'
    test_nas,test_seq,test_activity = parsedata(all_data, datatype)
    
    # Write the training and test sequences into separate fasta files. We will use these
    # for running RNAfold calculations.
    write_seq_to_fasta(train_nas, train_seq, 'train.fasta')
    write_seq_to_fasta(test_nas, test_seq, 'test.fasta')
    
    # Parse the predictions from RNAfold
    rnafold_train_outputfile = 'rnafold.train.1.fasta'
    rnafold_test_outputfile = 'rnafold.test.1.fasta'
    
    secstruct_vect_train, energy_train = parse_rnafoldoutput(rnafold_train_outputfile)
    secstruct_vect_test, energy_test = parse_rnafoldoutput(rnafold_test_outputfile)
    
    # Calculate features for training dataset
    train_features,fheaders = get_features(train_seq, train_activity, secstruct_vect_train, energy_train, pssm, train_nas)
    
    # Calculate features for test dataset
    test_features,fheaders = get_features(test_seq, test_activity, secstruct_vect_test, energy_test, pssm, test_nas)
    
    # Write the training and test features into their respective files
    np.savetxt(train_feature_file, train_features, delimiter=',',header=','.join(fheaders), fmt='%.3f', comments='') # Round to the nearest 3 decimal points
    np.savetxt(test_feature_file, test_features, delimiter=',',header=','.join(fheaders), fmt='%.3f', comments='') # Round to the nearest 3 decimal points
    

if __name__ == '__main__':
    main()