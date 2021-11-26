# fastpca
Fast principal component analysis for very high dimensional data implemented according to C. Bishop's book "Pattern Recognition and Machine Learning". For high-dimensional data, fastpca.m is much faster than Python's PCA function from sklearn and Matlab's PCA function.

Decrease in computation time results from calculating the PCs from the (smaller) covariance matrix of the transposed input-matrix "data" instead of the large covariance matrix of the original input matrix which are then use to project the observations to achieve the PCs of the large covariance matrix.
