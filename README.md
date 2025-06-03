# Protein ΔΔG Prediction

This project trains a model to predict ΔΔG values on protein-protein interaction upon single-point mutation.

## Dataset Format Requirements

The training data must be provided as a CSV file containing the following columns:

1. **pdb_id**  
   - Matches the PDB filename (e.g., `1ABC` corresponds to `1ABC.pdb`).  

2. **mutation**  
   - Follows the format `<WT><chain><position><MT>`. For example,  `AL123G` .

3. **ddG**  
   - The ΔΔG value (float) associated with each mutation.  
   - Used as the target label for model training.

Example CSV:
```csv
pdb_id,mutation,ddG
1ABC,AI123G,1.25
...

# Train
Run the training script with the following command:

'''
python train.py <csv_path> <pdb_dir>
'''

# example
'''
python train.py ./data/skempi.csv ./pdbs
'''

