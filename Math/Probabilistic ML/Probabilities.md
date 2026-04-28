It is all based on the Bayes' Theorem: $$p (x |y) = \frac{p(y|x) p(x)}{p(y)}$$
![[Screenshot 2026-03-02 at 15.13.18.png]]

```{python}
from jax import numpy as jnp
from jaxtyping import Array, Float

def Bayes(joint:Float[Array, "d_x d_y"])-> Float[Array, "d_x d_y"]:
	prior_X = jnp.sum(joint, axis=1)
	evidence_y = jnp.sum(join, axis=0)
	likelihood_for_X_of_Y = joint / prior_X[:, None]
	posterior_X_given_Y = (likelihood_for_X_of_Y * prior_X[:, None]) / evidence_Y[None, :]
	return *
```

![[Screenshot 2026-03-02 at 15.29.41.png]]

![[Screenshot 2026-03-02 at 16.03.13.png]]![[Screenshot 2026-03-02 at 16.35.33.png]]![[Screenshot 2026-03-02 at 16.37.39.png]]
Things to do: review the kolmogorov axioms for probability space + review the uniform and try to code a nice plot with that. 