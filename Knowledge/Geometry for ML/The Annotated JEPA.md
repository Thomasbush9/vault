Notes from blog: https://elonlit.com/scrivings/the-annotated-jepa/


The JEPA template: 
We start from semantically related views of the world (x, y, z): x is what we observe, y is what we want to predict and z is an optional latent capturing unknown factors. 

==The only structural requirement is that knowing x should constrain what y can be==. 

We encode both observations into a shared representation space:
$$s_x = f_\theta(x), s_y = f_{\bar{\theta}}(y)$$
Here the two functions are the encoders that can either be the same (for same modality ) or differ (for multi-modality)
The prediction happens in the representation space: 
$$\hat{s_y}= g_\theta(s_x, z)$$
And we train by minimizing the distance in the representation space between the prediction and the gt: 
$$L(\theta, \bar{\theta}, \phi) = D(\hat{s_y}, s_y)$$
