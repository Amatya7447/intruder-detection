import pycuda.driver as cuda
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np

# Size of the vectors
n = 1000000

# Initialize the input vectors
a = np.arange(n).astype(np.int32)
b = np.arange(n).astype(np.int32)

# Allocate memory on the device
d_a = cuda.mem_alloc(a.nbytes)
d_b = cuda.mem_alloc(b.nbytes)
d_c = cuda.mem_alloc(b.nbytes)

# Copy the input vectors from the host to the device
cuda.memcpy_htod(d_a, a)
cuda.memcpy_htod(d_b, b)

# Define the CUDA kernel function
mod = cuda.module_from_file("vector_add.cubin")
vector_add = mod.get_function("vectorAdd")

# Set the grid and block dimensions
block_dim = (256, 1, 1)
grid_dim = ((n + block_dim[0] - 1) // block_dim[0], 1)

# Launch the CUDA kernel
vector_add(d_a, d_b, d_c, np.int32(n), block=block_dim, grid=grid_dim)

# Allocate memory on the host for the result
c = np.empty_like(a)

# Copy the result from the device to the host
cuda.memcpy_dtoh(c, d_c)

# Print the result
print("Result:")
print(c)
