# Recurrent Neural Networks 

## Foundations of RNNs

We have three main components in our additive system (which is a particular example or RNN): the analog signal, the readout component and the external stimuli; plus a bias term. 
Some term to remember:
[[Hyperbolic Tangent tanh(x)]]

### Unfolding/Unrolling

So, as we have seen the network can be defined by this recursive equation : $\vec{s}[n]= W_s \vec{s}[n-1] + W_r \vec{r}[n-1] + W_x \vec{x} [n] + \vec{\theta}_s$ 
Where:
-  $W_s \vec{s}[n-1]$: is the signal inside the system
- $W_r \vec{r}[n-1]$: is the readout signal, a non-linear mapping of the signal. Usually tanh 
- $W_x \vec{x} [n]$: input terms 
- $\vec{\theta}_s$: bias term
The equation represents the system before the "unrolling", moreover we can ignore the effects of the state signal on the system memory, thus we have: 
$\vec{s}[n] = W_r \vec{r}[n-1] + W_x \vec{x} [n] + \vec{\theta}_s$
Which can be interpreted as that only the activations are stored in memory during the recursion. 
RNN essentially implements the backward Euler numerical interation method for the ordinary DDE. 

>Applying the RNN as a model implies that the underlying phenomenon is governed by the time differentiations equation (see above), whereby the role of the RNN is that of implementing the computational method for solving this DDE using the backward Euler discretization rule, under the restriction that the sampling time step $\Delta T$ is equal to the delay. 


==Finally the Unrolling:==

We unroll the standard equation by specifying the initial conditions on the state signal, and numerically evaluate the equation for a finite range of discrete steps indexed by $n$. 

As the equations describing the RNN are recursive in the state signal $\vec{s}[n]$, thus, the state signal at some value of the index $n$ encompasses the contribution of the state signal $\vec{s}[k]$, for all indices  $k<n$ , ending at $k=0$. Due to this attribute the RNN belongs to the category of [[Infinite Impulse Response systems ]], however in practice we approximate a sequence with an infinite support (IIR) by a Finite Impulse Response (FIR) 

## RNNs overview (cite paper)
Basically the paper will be an overview from the end of the previous paper. Thus, starting from the network itself towards the principles. 
### Backprop through time: 
