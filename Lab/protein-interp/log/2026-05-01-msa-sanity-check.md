Since we have noticed how the pair representation start at the step 0 kind of constant, we have decided to focus on the input msa and see whether it is different or represented similarly by Boltz-2. 

More analysis on the MSA and Boltz: 

- Check results with MT and WT MSA: copy wt msa, change the query line with each sequence [X]
- Check results WT and mutant MSA -> same process but for a random perturb sequence msa [X]
- random sequence + WT MSA -> same but first generate a random sequence [X]

Then we can start asking whether the hidden representation of boltz allows a classifier to reconstruct the WT sequence or the MT ? Here we have to think more about it. 


- p20 with artificial MSA (other p20 variants, gradient from p20)
- run WT-MSA + gradient of perturbations to see at which p the structure is still similar to the WT 
- run no-msa of point-wise mutations

Alternatives to the MSA: 
- Mutation aware msa 
- MSA dcontamination: remove rows above a sequence-identity with the WT 
- Protein language model embeddings-> how to add it to the pair representation? 


We can see a high tm-score (above pure noise) even with a random sequence 0.33. We could test until which perturbation level/same msa we obtain the a high score >0.5

### Experiments 

**Sequence perturb - MSA constant** 05/01:

- Goal: check how much you can perturb the original sequence using the WT-MSA and you still get a high tm-score 
- Input: 10 random perturbations of the input (p level) linked to the WT-MSA
- output: structure predictions + tm-scores
- Caveat: we want to use some n examples like 10 sequence per perturbation level

**Sequence constant (WT) - perturbation of MSA** 05/01:
- Goal: check how much you can perturb the original MSA using the WT-sequence and you still get a high tm-score 
- Input: 10 random perturbations of the input msa (p level) linked to the WT
- output: structure predictions + tm-scores
- Caveat: we want to use some n examples like 10 sequence per perturbation level

Experiments 05/04: 

- Run calibration scripts for Protforge [X]
- Read interpretability papers for Boltz 
- study more interpretability analysis [X]
- try to add the new nvidia boltz -> tomorrow 



