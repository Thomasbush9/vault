Since we have noticed how the pair representation start at the step 0 kind of constant, we have decided to focus on the input msa and see whether it is different or represented similarly by Boltz-2. 

More analysis on the MSA and Boltz: 

- Check results with MT and WT MSA: copy wt msa, change the query line with each sequence, add new path 
- Check results WT and mutant MSA -> same process but for a random perturb sequence msa
- random sequence + WT MSA -> same but first generate a random sequence

Then we can start asking whether the hidden representation of boltz allows a classifier to reconstruct the WT sequence or the MT ? Here we have to think more about it. 

Alternatives to the MSA: 
- Mutation aware msa 
- MSA dcontamination: remove rows above a sequence-identity with the WT 
- Protein language model embeddings-> how to add it to the pair representation? 


We can see a high tm-score (above pure noise) even with a random sequence 0.33. We could test until which perturbation level/same msa we obtain the a high score >0.5