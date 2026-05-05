1 group of cells -> extract the RNA -> expression information (genes that are produced) on avg. 
Single Cell RNA seq: one cell per droplet -> capture RNAs -> 


Notes from Nvidia Meeting: 

## review data:


Amplification: we amplify the expression + check the actual source

Identity fo the gene that is captured: alignment step-> taken all the reads and map them to all the genes. 
	 You create a references: 60k sets-> map to one of the 60k (genes)


Starting data: 
- Since we are adding a new gene, we add a new probe for that one. ]
- Another dataset for the results
- For each bar code we have a distinctive  features (how many time we saw that correlation)
- for each cell we also have a gene ID 


Barcode: count=> total count of genes detected and unique


Run MapMyCells-> map the identity of the cells 
Normalization + lognorm
check transcriptome 