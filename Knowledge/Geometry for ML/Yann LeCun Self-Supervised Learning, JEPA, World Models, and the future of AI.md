You search, given the input, for an output that minimise the energy function of compatabiliy between the two -> optimisation problems (not necessarily back prop.)

This system is not auto-regressive, as it has limitations; 
	- fixed numbers of computations (for complex tasks you trick the system to produce more tokens-> CoT)
	- Divergent process: you can't exactly predict the right word-> at any point the token generated could push you far from the right ones. 

We would like to build a world-model: given a representation of the state of the world at time t -> prediction of the representation of the state of the world at time t+1

The proposed solution is to predict a representation of the actual output (e.g., not at the pixel level). We may eliminate the details that are not predictable-> makes the prediction tractable (Joint embeddings)-> **JEPA**

The way to conceptualise the train of this system is using an energy based model that produces low energy for pairs that we observe and high energy for the one that we don't observe. 
	A flexible energy surface can take any shape-> we need to find a loss function that shapes the energy surface so that:
	- data points have low energies 
	- points outside the regions of high data density have higher densities. -> regularized methods. 
	- from energy to probabilistic: gibbs-boltzman conversion (but it's untractable-> just deal with energy)
