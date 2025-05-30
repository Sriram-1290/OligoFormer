@echo off
cd .\RNA-FM\redevelop
for %%f in (Hu_siRNA Hu_mRNA Mix_siRNA Mix_mRNA) do (
    python launch/predict.py --config="pretrained/extract_embedding.yml" ^
    --data_path=../../data/fasta/%%f.fa --save_dir=../../data/RNAFM/%%f ^
    --save_frequency 1 --save_embeddings
)
cd ..\..