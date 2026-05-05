It measures the similarity between two protein structures. It is intended to be more accurate than the RMSD measure (global metric). Its values range between (0,1] . How to read it: 
- 0.2 < : random chosen unrelated proteins 
- >0.5: they have the same fold. 

It is defined by:

$$\text{TM-score}= max [\frac{1}{L_{target}} \sum_i^{L_{comm}} \frac{1}{1 + (\frac{d_i}{d_0 (L_{target})^2})}]$$
Where: 
- L target is the length of the amino acid sequence of the target protein,
- L common: is the number of residues that appear in both template and target structures 
- $d_i$ is the distance between i-th pair of residues 
- $d_0 (L_{target})^2$ = $1.24 \sqrt{L_{target}-15}^3 -1.8$  is a distance scale that normalises the distances 
When comparing two protein structures that have different sequences and/or different residue orders, a [structural alignment](https://en.wikipedia.org/wiki/Structural_alignment "Structural alignment") is usually performed first, and TM-score is then calculated on the commonly aligned residues from the structural alignment.

More info at the ![Wikipedia page](https://en.wikipedia.org/wiki/Template_modeling_score)


