# Connected Components
### ccpt
###

Given a graph of objects and a set of edges [rowid_0, rowid_1], where a row in 'edges' draws an edge between two object, and where each rowid in 'edges' indicate a row and therefore an object's information in 'pts', we find each connected component. 

This library is implemented using the jit-compiled CUDA API built into Python's Numba. I've translated the kernels that implement this algorithm into python-like CUDA Numba, more or less one-to-one from another implementation in native CUDA.

~~~~
[Todo: find the original author on github and link here]