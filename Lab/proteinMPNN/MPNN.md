What if we add to MPNN the MSA knowledge? or some sort of knowledge from the msa? 

[[Yann LeCun Self-Supervised Learning, JEPA, World Models, and the future of AI]]
Would it make sense to make the prediction in the embedding space? So we have: 
- Encoder: process the inputs to extract features-> 3d backbone + info? 
- Predictor: takes the context embedding and a latent variable to predict the target embedding 
- Target encoder: slowly moving avg of the main encoder-> stable targets. 

In this way we would not have to restrict our predictions to the aa level which is trickier? we could add additional information like msa/energy info in the embedding layer, we won't have the AR issues.