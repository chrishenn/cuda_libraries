<h1 align="center" style="margin-top: 0px;">Cuda Libraries</h1>

<div id="img0" align="center">
    <img src="doc/images/graph2.jpg" height="300" width="500">
</div>


An abridged collection of CUDA and Pytorch C/C++ extensions. Each provides some basic transformation on GPU data as part of an ML model's data flow. Many make use of the Torch C++ API for easy CMake compilation and incorporation into a PyTorch model training loop. Some older libraries use the Ninja compiler built into python's buildtools.

Given that these libraries were each a small part of some architecture concept under rapid investigation and prototyping, they include no tests. 