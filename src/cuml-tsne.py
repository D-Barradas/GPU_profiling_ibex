# demo for profiling python using command
#
# nvprof --print-gpu-trace python cuml_tsne.py

from sklearn.datasets import load_digits
X, y = load_digits().data, load_digits().target
from cuml.manifold import TSNE
tsne = TSNE(n_components = 2)
X_hat = tsne.fit_transform(X)

# code is based on https://medium.com/rapids-ai/tsne-with-gpus-hours-to-seconds-9d9c17c941db
# and needs sklearn and cuml, which are both available for ppc64le in conda channels. 
# 
