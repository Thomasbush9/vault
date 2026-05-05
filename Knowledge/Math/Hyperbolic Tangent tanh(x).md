It is a common activation function of the form:
$$\begin{equation}
\frac{\exp(z)-\exp{-z}}{\exp{z}+\exp{-z}}
\end{equation}$$
Which result in the following shape: 


```functionplot
---
title: Tanh(x)
xLabel: 
yLabel: 
bounds: [-1.5,1.5,-1.5,1.5]
disableZoom: false
grid: true
---
f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```
It is used as it has some nice properties:
- [[Monotonic]] 
- [[Negative Symmetric Function]]
- [[Quasi-Linear]] region
- [[Bipolar saturating ]]
