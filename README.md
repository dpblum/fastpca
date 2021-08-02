# fastpca
Fast principal component analysis for very high dimensional data implemented according to C. Bishop's book "Pattern Recognition and Machine Learning", p. 570. For high-dimensional data, fastpca.m is much faster than Python's in-built PCA function (from sklearn.decomposition) or MATLAB's in-build function pca.m. 

Decrease in computation time results from calculating the PCs from the (smaller) covariance matrix of the transposed input-matrix "data" instead of the large covariance matrix of the original input matrix which are then use to project the observations to achieve the PCs of the large DxD covariance matrix. 
 
By default, fastpca removes the mean of each observation.  In this first implementation of fastpca, I skipped calculation of Hotelling T-Squared Statistic as I didn't need it so far. 

INPUT: 
data = matrix of size n*p
n    = number of observations
p    = number of dimensions

OUTPUT:
c    = principal components, PC
sc   = PC scores (projections of observations on PC's)
lat  = variance captured by each PC score
expl = cumulative lat in percent
The p-n PC's that does not explain any variance 
are automatically excluded from the results
