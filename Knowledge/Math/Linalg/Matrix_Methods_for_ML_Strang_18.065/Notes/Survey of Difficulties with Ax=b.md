### Matrix Nearly Singular:
Elimination would give a poor result.
- SVD: it would give an answer in this case $A= U\Sigma V^T$ and $A^{-1}=U^T \Sigma^{-1} V^T$
But a better solution would be:
- Using $l_2$ norm, minimize: 
$$min ||Ax -b ||^2 + \delta^2 ||x||^2$$
Where $\delta$ is bigger than 0:
We can rewrite the system as: 
$$\begin{pmatrix}
A \\
\delta I
\end{pmatrix}

\begin{pmatrix}
x
\end{pmatrix} = 
\begin{pmatrix}
b \\
0
\end{pmatrix}
$$
We can rewrite the system as the equation that we are trying to minimise. ==Read how delta should be set to==

- If $\delta$ is set to 0, then the solution of the system goes to the pseudo-inverse (check the proof and code a simulation)
It is a possible solution for when we get the output of a system and we want to obtain the full system that has produced the output. 
For any matrix $A$:
$$(A^T A + \delta I)^{-1} A^T \rightarrow A^+$$
Which is the pseudoinverse. ==Review statistic approach to l1 norm==
Prove the theorem for 1x1 and 2x2 using the SVD. 