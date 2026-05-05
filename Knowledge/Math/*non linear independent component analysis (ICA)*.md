Suppose that you have a recording with many voices, then you would have a mixture of independent sources. ICA aims to separate these sources using the following assumptions:
1. each observed signal (readout of sources) is a linear combination of independent sources
2. The sources are statistically independent and non-Gaussian 
Basically,  given $x=As$ where x is the observed vectors, A is the unknown mixing matrix and s is the unknown source vector. ICA learns a matrix W to unmix the observations: $s = Wx$ . 

The central issue is that you can't just invert the function, as often it is not linear. 

**Modern View**:

A common idea is to assume that the sources vary over time or depend on auxiliary variables. The main ideas are:
1. Time structure: each latent source follows a nonstationary or time-dependent process. If two sources change independently over time, then they must be separate-> NN can solve this issue with a contrastive loss, like [[infoNCE]] 
2. Auxiliary Variables: assume that the distribution of the sources depends on some observed variable (e.g., a label or context variable u): $p(s|u)$ then you train a model $f^{-1}$ that maps x into s, ensuring the estimated s has variations that depends on u, while being as independent as possible otherwise. 
3. Contrastive Learning: Basically we train a network to distinguish between different time segments or conditions, encouraging the network to learn representations that separate the sources.  

## Comparison with T-sne and UMAP:

The latter two methods are not identifiable, as you are not trying to recover the original latent cause they just aim to visualise the embeddings. 