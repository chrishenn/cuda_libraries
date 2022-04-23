# Connected Components
### ccpt
###

Given an Nx2 set of edges [[id_0, id_1], ...], where a row in 'edges' specifies an edge between objects with unique ids id_0 and id_1, we find each connected component in the graph. 

This library is implemented using the jit-compiled CUDA API built into Python's Numba. I've translated the kernels that implement this algorithm into python-like CUDA Numba, more or less one-to-one from another implementation in native CUDA ([jyosoman/GpuConnectedComponents](https://github.com/jyosoman/GpuConnectedComponents)), citing:

    Fast GPU Algorithms for Graph Connectivity, Jyothish Soman, K. Kothapalli, and P. J. Narayanan, in Proc. of Large Scale Parallel Processing, IPDPS Workshops, 2010.

The original code is also included here for reference under ccpt/GpuConnectedComponents-master.