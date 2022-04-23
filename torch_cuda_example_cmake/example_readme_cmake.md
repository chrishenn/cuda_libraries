## Readme for building cuda and torch-c++ extensions with cmake

This example extension builds (with cmake) under torch 1.6.0, libtorch 1.6.0, cuda 10.2 and cudnn 7.6.5.

Everything needed to build with cmake, except the libtorch folder, is installed into /om2/user/chenn/torch-gpu-cmake.simg, which 
should be accessible to everyone.

    NOTE: cuda 10.2 will not run on openmind without a singularity container. 
    
    NOTE: cmake will automatically detect all device architectures available on a machine (or cluster) and compile device code for all of them. 
    
    NOTE: building with cmake is required to use an extension with torchscript.
    
    NOTE: the method for loading extensions that are built with cmake seems much more robust than with ninja - the torch doc also hints that it will
          be better supported, though of course the doc is unreliable
          
    NOTE: see example_python_call.py for the (distinct from ninja-style compilation) syntax to load and call into this library
    
    NOTE: the header needed in example_define.cpp must be torch/script.h for cmake, rather than the torch/extension.h that we
          used for building with buildtools/ninja. The header in example_driver.cu can be either torch/types.h (as we used for ninja), or torch/script.h.


## Openmind:

I believe everyone can use my folder on /om2/user/chenn/libtorch - this path is in these example CMakeLists.txt files.

Next, add singularity module to running session:

    module load openmind/singularity/3.5.0

Here we're using my existing container on /om2/user/chenn. We need a custom singularity container because openmind does not have a 
prebuilt module with the correct versions of cmake.

Head to the extension folder and create a build/ subdir (if there is none). From inside build/, run:

    singularity exec --nv -B /om,/om2 /om2/user/chenn/torch-gpu-cmake.simg cmake ..
    
Then:

    singularity exec --nv -B /om,/om2 /om2/user/chenn/torch-gpu-cmake.simg make -j /n-threads/

 


## Local-Machine installs

Make sure cmake is installed on your system, and cuda 10.2, and torch 1.6.0 is in your active python environment. See ooenv1.yml for a complete working
environment.

Install cudnn 7.6.5 from .deb files for both cudnn and cudnn-dev, as:

    dpkg -i <file>
    
for each .deb file. you'll need an nvidia developer acct. I've had good luck with the .deb installs though there are other installers.

Download and extract libtorch, and make note of its path. I used

    wget https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.6.0.zip
    
to download. 

Then, modify the "CMAKE_PREFIX_PATH" in the CMakeLists.txt file to point to your libtorch folder. 

Or, you can override the path in the CMakeLists.txt by passing this flag with the cmake command, as in:

    cmake -DCMAKE_PREFIX_PATH=[absolute-path-to-libtorch-folder] ..

in project folder:

    mkdir build 
    cd build
    cmake ..
    make -j [n-threads]
    
The binary is put in the build/ dir and has 'lib' tacked onto the beginning of the filename.





