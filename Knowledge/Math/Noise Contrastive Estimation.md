it is a method for estimating parameters of a statistical model. the key idea is to run a logistic regression to tell apart the target data from noise (you could add the application on word embeddings). 

Considering $x$ as the target sample $\sim P(x|X-1;\theta)= p_\theta(x)$ and $\hat{x}$ be the noise sample $\sim P(\hat{x}|C=0) = q(\hat{x})$.   