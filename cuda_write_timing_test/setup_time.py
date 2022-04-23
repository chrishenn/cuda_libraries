from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
	name = 'timings',
	ext_modules=[
	    CUDAExtension(
			name='time_cuda',
			sources=['frnn/time_ext_define.cpp', 'frnn/time_driver.cu', 'frnn/CUTime.cu'],

	        extra_compile_args = {'cxx': ['-g'], 'nvcc': ['-Xcompiler', '-rdynamic', '-lineinfo', '-prec-sqrt=false']}
)
	],

	cmdclass={
	    'build_ext': BuildExtension
	})
