## First Analysis
So it seems that these models struggle with designing structures that reflect the impact of mutations. As a first step it would be interesting to check that they are actually memorising some amino acids for that protein specific. 

The main concern is that AF has learned just to do template matching rather than learning biophysical properties. 

## Paper on Adversarial Sequence Mutations in AF: 

**Experiment 1**
They run AF3 on 200 proteins sequences (monomer, multimer, similar and novel) on 5 level of mutations 5, 10, 20, 40 and 70% and computed the TM-score and DockQ to measure the similarity to the original protein. 
![[Screenshot 2026-03-05 at 10.21.58.png]]

We can see how AF3: 
- Maintains an high similarity even with 40% mutated sequences
- AF3's internal confidence shows only moderate sensitivity to perturbations 

**Experiment 2**

They run the same test using 45 experimentally validated proteins, targeting aa that have been show to induce fold-switching. However, the observe the same results as with the previous test. 


**Experiment 3: residues deletions**

They run the same test, but using residues deletions (which have been shown to be more deleterious to protein stability), using 1, 3, 5 and 10% levels. They observe a similar patter, with AF3 producing similar structures even with 5% deletions and with high confidence 

![[Screenshot 2026-03-05 at 10.29.54.png]]

**Comparison with ESMFold**

They compare AF3 to ESMFold (do not use MSA, only supports monomeric):  They see how fro point-mutations ESMFold is more sensitive to mutations than AF3

![[Screenshot 2026-03-05 at 10.32.06.png]]


They also observe how the confidence of AF3 is correlated with structural (template) similarity rather than sequence similarity in the training data. 

## RSA of Protein Models 
There are studies like the one before that focuses on studying the models behaviour on different stimuli, but as we do with neuroscience it would be interesting to study how the internal representations changes when the model behaves in a specific way. 

We can use RSA to study how does mutation change the internal geometry of representations across layers. If, as this paper suggests, these models are invariant to mutations, then we should see a correction of the different input somewhere in the model; on the other hand, if the models are able to capture differences between mutated sequences (cite effective strain paper) we should see a dissimilarity between their internal representations. 

We should use some metrics to measure the output similarity: 
- [[TM-score ]]
- Effective Strain on multiple runs 
- RMSD
- Confidence 
Then we also measure the similarity of the hidden layers. We should also add the same analysis intra-sequence and between iterative steps. 

Techniques that we can use: 
- [[CKA: Centered Kernel Alignment]] :
- RDMs: Representational Dissimilarity Matrices: extract the embeddings for a set of sequences + mutants for each layer. Compute pairwise distance at each layer. Compare the RDMs across layers and across inputs. 

The overall goal is: given a set of proteins of which we know a set of mutations that lead to changes in the function/sequence, how these models represent them across layers? Is the structure similar or not? if the models are invariant to these mutations, where these alignment happen in the models? 
### Points to Think About: 

We have to consider how these representations relate to proteins as they have a stricter geometry compared to images.


### How does structural information emerge across layers:

- At each layer correlate layer_s RDM with the alpha carbon matrix (long-range only). 
- Layer wise CKA with final layer: how similar is each layer to the final output where the representation converges on the structural solution. 

**Mutations**

Given a WT and mutant sequences, we can compare layer-per-layer representations to find where in the network mutations cause divergence. 
- Per-layer delta representations: we compute the distance between WT and mutant per each layer, if the model actually has a predicts a different structure it should diverge through the layers. 
- Per-residue perturbation profile: what are the residues representations that change the most between pairs. 
- Layer-wise delta RSA
If mutations produce the same final structure but take different representational paths through the layers. 

### Working Plan: 

Create the Dataset:  
- Sample $N_{e}$ and $N_n$ from the GFP dataset, based on the values of fluorescence + add a small sample of random mutations for the e group where you mutate higher proportion of the sequence (follow paper up to 70%?)
- Generate a .fasta directory with all the sequences [X]
- Run MSA pipeline for all the sequence-> check the msa if it changes
- Run the Boltz with the hooks on the sequences (save for each recycling step) + ESMFold, AF, Simple fold + 
Analyse the results: 
- For each structure compute the comparison with the WT: TM-score, RMSD, Effective strain?
- Analyse how each model represents the mutants vs. the WT
- high effect mutants vs. neutral
- If there are differences between check how the representations are across models x layers 
- Can we use these representations to predict the effect? or can we improve these models? 

Studies on the ability of these models to correctly predict mutations effect have been done, but they focused on external metrics (TM-score, effective strain), it would be interesting to understand where these models converge in their hidden representations etc. 

The appeal of this approach is that we are not restrict to a single architecture, but we can potentially run the same analysis for any model, as long as we can extract the hidden representations and for new inputs. 

It can also be used to validate new models etc. 


To analyze how different inputs are represented within a single model, I used a representational similarity approach across samples rather than across layers. For each layer of the model, I extracted a representation for each mutated protein sequence (e.g., by pooling residue-level embeddings or selecting the embedding at the mutated position), producing one vector per sequence per layer. I then computed a pairwise similarity matrix between all sequences at that layer, capturing how similarly the model represents each pair of mutations. By comparing this similarity matrix to a hypothesis matrix encoding class membership (neutral vs. effect), I quantified whether the geometry of the representation space aligns with the biological classification. Repeating this procedure across layers allows identification of where in the model hierarchy the distinction between neutral and functional mutations becomes most pronounced, and whether this separation strengthens or weakens across depth.

Better approach: 

Use the wildtype vs. harmful mutations for each layer. Then measure teh distance between the two for each layer. You can correlate to the phenotype, do it for both mutatation classes.

You’re asking:

> Does the model’s internal geometry reflect functional consequence, or is it invariant to it?

And you’re testing that not by arbitrary clustering, but by **comparing each mutation’s representational displacement from wild type to its measured phenotypic impact**.

That’s a strong, interpretable, and publishable approach
## Recommended pipeline (clean and defensible)

1. **Within-model analysis (RSA / distance-to-WT):**  
    Identify which layers encode mutation effect.
2. **Across-model analysis (CKA):**  
    Identify whether models learn similar or different internal geometries.
3. **Joint interpretation:**  
    Relate differences in representation to biological performance or invariance.
if we see that the model is senstive at some layer l we can also inject the repreentation in later layers 


Next step: review how boltz's architecture is and start thinking from where to extract the hidden representations, ESM model. 

Another interesting approach would to show different perturbations at different input/hidden representations. 

Now we have saved all the hidden representation from all the types, our dataset is now composed of:
- 100 high effect multi mutations:
	- hidden representations 
	- MSA
	- best structure 
100 neutral multi mutations:
	- hidden representations 
	- MSA
	- best structure 
1 original:
	- hidden representations 
	- MSA
	- best structure 
Next step: 

1. develop pipeline to confront the structures: [X]
2. pipeline to construct confront between and intra models
3. we should also compare the input embeddings and how they change 
4. compare the attention scores through the triangular attention updates 
5. Try to run same analysis without msa 
6. Run the same analysis with 100 highly mutated **sequences**
7. try with different sequences
8. what if we study the same perturbations with different kind of models, such as LLMs?∆
Check also other metrics, Singular values comparison


Repeat the analysis but using a fixed window going across the protein (maybe exclude the last 20aa)

Run on a new protein synthetic (0 similarity with the whole dataset, but with very high confidence )

For the mutations we can compute the BLOSUM score for each mutation-> average/max etc. 


Experiments 04/16:
- produce the augmented generated data with p20, p40, p70 perturbations [X]
- launched synthetic data generation [done]
- Run structural scores [X]
- Obtain also confidence outputs [X]
- Run hidden representation extractor for Bolts: s at every layer, attention at different steps, pairformer only on few layers? 
	- same as before [Done]
	- Attention and pairformer for selected layers [done for 20 %]
- Focus on possible points of contacts or attention patterns as they could be more informative about the mutation effects. 
- Start implementing the job for ESMFold (add it to the pipeline etc.)+ hidden representations  [done]
- What about synthetic proteins [done]
- Plan future experiments 
What if we also perturb or simple cut the msa for the mutants? 

Experiments 04/17: 
- ESMFold: 
	- Make a run with a few fasta to check it [X]
	- Batch run with slurm scritp [X]
	- Make it supported by ProtForge [X]
	- Add hooks to ESMFold 
- Interpretability on perturb proteins: 
	- Study metrics for similarity on attention: first find a scoring function for each aa and use it etc. 
	- Compute the divergence using just the sequence embedding 
	- Compute attention scores overall 
- Synthetic data: 
	- Generate perturb classes and run usual pipeline 
- Study: 
	- Continue with stats for diffusion 
	- ML course 
	- write the results of the experiments etc. 

Experiments 04/19:

- Data Generation: 
	- generate a dataset of mutations + perturbation to use 
	- Perturb the synthetic proteins: 10, 20, 40, 70 perturbations for 50 sequences each. 
- ESMFold: 
	- Compute similarity metrics for the esm fold -> to do 
	- study how to add hooks to esm
- RSA:
	- Use better metrics for attention similarity 
If the goal is:

> “understand whether the patterns are similar and where they are conserved”

then my recommended core set is:

- **Spearman correlation** → same ranking of important pairs?
- **Top-k overlap** → same strongest residue-pair interactions?
- **JS divergence row-wise** → which residues changed their interaction profile?
- **CKA on pair representations** → which layers stay invariant overall?

That combination is stronger than just Frobenius/L2

To check whether the model does any sort of template-matching, anchor tokens, we can take the distributions at p40 for example, check the examples where the tm-score is high and the one where it's low, check for patterns that are conserved in the first case, clamp them and increase perturbation a bit or reverse the others.

Experiments 04/20:

- ESMFold:
	- Finish structural scoring for perturbations 
	- Extract hidden representations 
- Boltz:
	- Analysis of pairformer + attention 
	- Run divergence for sequence 
- Compare boltz and ESMfold representations if you notice differences in the outputs 
- Start analysis on synthetic proteins 


In the pairformer we can observe a first high similarity because the residues are exploring the local relations so the attention effect is uniform regardless of their idetity, then there is a drop and a final recover 


It seem that during the first step Boltz encodes some of the noise in the mid-layers and then the representations become constantly more similar to the original. While the pairformer has a drop at the last layer (however the previous representations remains constant). Surprisingly ESMFold seems to capture more the perturbations even at 20% level. 

Experiments 04/21
At this point it would be interesting to: 
- Run the distance metric at the attention level for all the perturbations at .4 with the original to have a better metric [X]
	- Do the same for all the steps and both perturbations [X]
	- how does it look across steps and at the two levels of perturbations -> it follows the same pattern [X]
	- It seems that the layer_z remains the most constant: drop at the last layer what if we pertub that while maintaining the input protein constant? [to-do]
- Do the plot for the confidence as well 
- Extract the hidden representation of ESMFold for the same inputs and start to compare their hidden representations -> we should observe an higher similarity in the mid-layers? 
	- p20 [X]
	- p40 [x]
	- Run cosine similarity for steps 0, 4 for p20 and p40 [X]
- Explore the synthetic proteins 
- Run the same analysis with the synthetic proteins 
- clean notebook and add functions
- re-run comparison structure sim with the original from esmFold to double check (also Boltz ones) [X]
- Run CKA between ESM-Fold and Boltz, can we align Boltz to hidden dims of ESMFold? 
- Since both ESMFold and Boltz share the same trunk structure we could change some representations?, otherwise the alignment option remains available. We could consider to align using CKA between models and then test it using the effective strain 
Distributed Alignment Search (localising causal variables subspaces with sgd)

- Rename all the sequences using bash [X]
- A possible interpretation could be that the esm starts with a similar representation (kind of random)-> builds a difference but then it keeps it across the recycling steps, while Boltz does not. 

The issue is that ESM seems to have a more aligned representation between WT and the perturbations, however the model produces outputs that seem more affected by the perturbations. 

Weekends Epxeriments 04/23:
Try to build a better metric rather than simple cosine similarity: 
We should extract the attention weights to actually check the similarity using the triangular attention
- Add hooks for the attention weights 
- Run the same analysis 
- take a look at all the structures 
- run it with the random protein 
- double check recycling loss
- compare the first step with the tm-score
- Interpretability from the last pairformer to the structure 
- train a model to recover information 
- check the second protein 
- check with msa and without 
- We could also perturb the diffusion input? 
- repeat analysis with the synthetic proteins-> just score them


Probably it would be interesting to have a negative example such a proposed sequence 


I think that it's worth to first try to investigate more how the diffusion model is dependent on the conditioning module. Continue to compare the hidden reps and add them to the general pipeline but zoom more on the diffusion module 

Meeting points: 

- Continue interpretability on Boltz-> try to understand where Boltz representations become similar to the WT regardless of the mutations
- Continue working on the pipeline for the protein design 
- What if we use something different from the MSA? in order to exclude the protein family from that 
	run analysis on the msa to check how similar they are etc. 
- What if we develop a learnable metric instead of tm-score or effective strain
- try with completely different input sequence but same MSA 
	- What about same input sequence but input msa from the first recylcin step
- Check discrete flow mapping for proteinMPNN model 

[[MSA Sanity Check]]

At the moment we have seen that the msa for the mutants is basically a subset of the original msa. Thus, we should start to think about how to change it to improve the model


Experiments 04-28,29
- Check the confidence 
- Run msa analysis over all the msa that we created [X]
- Generate structures for the msa free sequences [X]
		- generate dirs with cleaned msa[X]
		- remove the msa from the .yamls [X]
		- launch predictions [X]
- Compute the scoring: 
	- msa-free [X]
	- synthetic proteins 
- Try to perturb also the msa? remove columns? 
- Would be possible to use something different from the msa as our input? 
- Code uniprot scritp to generate the .fasta files for the protForge pipeline 
- Add hidden representations to the protforge 



 Experiments to do: 

- Check attention/representation for the msa module in Boltz
- Think about possible perturbations that we could add to the msa to make it more interesting to use in these setups 
- conservation vs. unique region for memory: we should look at the attention map in the msa/pair representation to understand which aa are the most informative 


Verify the current updates with codex + tests. 


