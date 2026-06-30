# Notes on Interpretability 



## OV circuit analysis 


Since we process the inputs using multi-head self-attention, we can rewrite the value and out projections partitioned by each head, by: $W_v^h \in \R^{d_k, d_{model}}$ which indicate the rows of the weights matrix $W_V$ and $W_O^h \in \R^{d_{model} d_k}$ the columns of the weight matrix $W_O$. We can then divide these projection by the head that they process, given the concatenate vector $z_h$: 

$$W_O W_V z = \sum_h W_O^h W_V^h z_h$$ 


Given this definition we can decompose the attention mechanism into two circuits: $QK$ and $OV$

- Value of each key per head: $v_k^h = W_V^h LN(x_k)$
- The attention weighted sum: $o_k^h = \sum_{k<q} a^h_{qk} v_k^h$
- Lastly we concatenate the results and project: $\delta x_k = \sum_h W_O^h o_k^h$

By putting all together we can rewrite the output of the multi-head attention layer as: 

$$\delta x_k = \sum_h \sum_{k<q} a^h_{qk} W_O^h W_V^h LN(x_k)$$

From this we can define the OV matrix as $W_O^h W_V^h \in \R^{d_{model}, d_{model}}$, which is applied to each attended token to produce the write vector: $w_k = OV^h LN(x_k)$. Finally, each attention head contribution can be defined as: $\delta x_k^h= \sum_{k<q} a_{kq}^h w(x_k)$. This decomposition, allows us to have the scalar term $a_{qk}^h$ that informs us about where the model attends for that token, and the OV circuit $w$ that tells us what to write. 


Check if other heads influence during steering

