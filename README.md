# Skin2
## Dataset:
The ISIC-2019 dataset used for model training can be downloaded from https://challenge.isic-archive.com/data/#2019.
### Method Algorithm:

Algorithm 1 Joint-Loss Function Optimization Algorithm.
1:Input:" Training set " I^i ",the last " FC" layer weights " W∈R^(d*n) ",learning rate " η",hyperparameter " α_t,β.
2: Output: W
3: while not converged do.
4: t←t+1
5: Compute the first loss L_t=E^2 (w^T w)
6: Compute the second loss L_c=E((w^T w)^2 )
7: Compute the joint loss L_"joint " =L_*+α_t L_c+βL_t
8: Calculate the backpropagation process  (∂L_"joint " )/(∂w_i^t ) " foreachiby "  (∂L_"joint " )/(∂w_i^t )=(∂L_*)/(∂w_i^t )+(∂L_c)/(∂w_i^t )+(∂L_r)/(∂w_i^t )
9: Update the parameter w_i " by " w_i^(t+1)=w_i^t-η (∂L_"joint " )/(∂w_i^t )
10: return

Bilinear CNN using a matrix-similarity based joint loss function for skin disease classification.
