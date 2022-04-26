# Connected Components
### ccpt
###

Given a graph of objects and a set of edges [rowid_0, rowid_1], where a row in 'edges' draws an edge between two object, and where each rowid in 'edges' indicate a row and therefore an object's information in 'pts', we find each connected component. 

This library is implemented using the jit-compiled CUDA API built into Python's Numba. I've translated the kernels that implement this algorithm into python-like CUDA Numba, more or less one-to-one from another implementation in native CUDA ([jyosoman/GpuConnectedComponents](https://github.com/jyosoman/GpuConnectedComponents)), citing:

    Fast GPU Algorithms for Graph Connectivity, Jyothish Soman, K. Kothapalli, and P. J. Narayanan, in Proc. of Large Scale Parallel Processing, IPDPS Workshops, 2010.

The original code is included here for reference under ccpt/GpuConnectedComponents-master