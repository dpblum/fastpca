import time
import numpy as np
from fastpca import fastpca 
from sklearn.decomposition import PCA

# Sample data (e.g. 300 observations with 500.000 features each)
data = np.random.rand(300,500000)

# Standard PCA
tic   = time.time()
pca   = PCA(); 
pca.fit(data)
c     = pca.components_
toc   = time.time()
t_pca = round(toc-tic,2)

# fastpca 
tic               = time.time()
cf, sc, lat, expl = fastpca(data)
toc               = time.time()
t_fastpca         = round(toc-tic,2)

print('standard PCA time:' + str(t_pca))
print('fastpca time:' + str(t_fastpca))
