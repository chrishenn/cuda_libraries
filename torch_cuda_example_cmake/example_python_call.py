import os

import torch as t
t.ops.load_library(os.path.split(os.path.split(__file__)[0])[0] + "/torch_cuda_example_cmake/build/libexample_cuda.so")


def example_python_call():
	input = t.randint(100000, size=[100], dtype=t.int).to(0)

	input, output = t.ops.example_op.example_kernel(input)

	if input.eq(output).sum() == input.size(0):
		print("success")
	else:
		print("failure")


if __name__ == "__main__":
	example_python_call()