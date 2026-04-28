If $B$ has rank $k$ then $||A-B|| \geq ||A - A_k||$ where $|| ||$ can be any of these three norms:
- $l_2$ norm: 
	- $||A||_2 = \sigma_1$ 
- Frobenius norm
	- $||A||_F = \sqrt{|a_{11}, ..., |a_{nn}|}$
- Nuclear norm
	- $||A||_N= \sigma_1 + ... + \sigma_n$
**Proof**
All the norms depend only on the singular values. Multiplication of A by orthogonal matrices does not change the norms. 
*PCA and Least Squares*
They are not the same thing. In least squares you minimise the errors, while in PCA you are measuring points perpendicular to the line. 
- PCA: involves SVD
- Least Squares: $min|b-Ax|^2 = A^TAx= A^T b$ which is the normal equation. 
Sample Covariance Matrix: $\frac{AA^T}{N-1}$ , we want to find the best line that connects the two dimensions ($\sigma_1$). 