# Fixed-Radius Nearest Neighbors
### frnn
###

Given an array of objects situated in a batch of images, we attempt to find objects whom can be paired together, according to some criteria. That criteria may include a function that projects object sizes into nonlinear space for comparison, and may compare objects' sizes or locations for pair-ability. 

'Fixed' may then sound like a misnomer; however, the goal of these comparison functions is to provide behavior that is invariant in one or more ways, in the context of training an ML model. 

Frnn folders with 'bipart' in the name only return pairs that match accross two partitions of a bipartite graph.

'Brute' generally denotes a simpler code design, where n^2 comparisons would be done for objects in each image. This would provide worse performance when the worst-case object-density is quite high, or better performance when the overhead of building intermediate spatial binning structures would outweigh the comparison time.

'frnn_kernel_designs' provides some possible kernel designs that would provide various performance tradeoffs in varied input scenarios. For relatively sparse input graphs, a brute-force method is often fast enough.

'frnn_cpu' includes a cpu-based implementation of frnn, since it is (was?) pytorch standard practice to provide device-agnostic code for model training. I used the @njit decorator to pass the function to Numba's "no-python" jit compiler, which provides a significant speedup over serial python iteration, somewhat optimizing and optionally parallelizing the work on cpu. Nevertheless, the space complexity is much worse due to Numba's inability to dynamically allocate tensors while computing the output (or provide a globally-atomic way to lock writes to an output structure). 

The structure of the code files in each library may vary depending on whether that library is built with CMake or Python's Builtools-Ninja. The Ninja compiler (and to some extent, CMake and the pytorch utility for importing precompiled functions) provides some unusual restrictions about where and how kernels can be compiled and included into a library (and become available to python). Therefore, I've had to use some inelegant code layouts to keep functionality separate and legible.  