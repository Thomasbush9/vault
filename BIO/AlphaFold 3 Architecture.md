Model type: [[Conditional Diffusion Model]] 

## Main Overview: 

1. Tokenization: each amino acid residue and nucleotide correspond to one token. 
2. The initial trunk feature is obtained by a feature embedder that performs attention over all atoms-> encode information about the chemical structure 
3. Pair representation + single representation -> conditioning network (heavy part of the computations)
4. Conditioning network: 
	1. Template Embedder: encoders information about the the provided template (not necessary) in the pair representation 
	2. MSA Module: it makes the MSA communicates with the pair representation 
	3. Pairformer: pair representation + single representation 
5. The outputs of the pairformer are then used to condition the diffusion network 
6. Diffusion Module: encodes denoising as a sigle step
7. Confidence Head: uses the pair representation + structure (predicted by the diffusion module) to provide confidence metrics

## Input Embeddings 

