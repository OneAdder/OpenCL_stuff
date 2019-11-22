import numpy as np
import pyopencl as cl
import os

# if it will say that the device is wrong, delete this line
os.environ["PYOPENCL_CTX"] = ""

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

prg = cl.Program(ctx, """
    __kernel void tensor(
        const unsigned int size,
        __global const float *m1,
        __global const float *m2,
        __global float *res) {
            int i = get_global_id(1); 
            int j = get_global_id(0);
            res[i + size * j] = m1[j] * m2[i];
    }
""").build()

def tensor_product(vec1, vec2):
    """Tensor product using OpenCL
    
    Parameters
    ----------
    vec1: class 'numpy.ndarray'
        First vector. Example: np.array([1, 2, 3]).
    vec2: class 'numpy.ndarray'
        Second vector. Example: np.array([1, 2, 3]).
    
    Returns
    -------
    res_np: class 'numpy.ndarray'
        Example:
            array([[1., 2., 3.],
                   [2., 4., 6.],
                   [3., 6., 9.]], dtype=float32)

    """
    # reshape np.array([1, 2, 3]) -> np.array([[1,], [2,], [3,]]).astype(np.float32)
    a = vec1.reshape(vec1.shape[0], 1).astype(np.float32)
    b = vec2.astype(np.float32)
    
    # copy the arrays to the graphics memory
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    
    # generate zeros array that has tha shape of resulting matrix
    r = np.zeros(shape=(a.shape[0], b.shape[0])).astype(np.float32)
    # create write buffer for the resulting matrix in the graphics memory
    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, r.nbytes)
    
    # run the prg shader
    prg.tensor(queue, r.shape, None, np.int32(len(a)), a_g, b_g, res_g)
    
    # create a variable for the result
    res_np = np.empty_like(r)
    # move the result from the graphics memory to the system memory
    cl.enqueue_copy(queue, res_np, res_g)
    return res_np

if __name__ == '__main__':
    # just check that the result is the same as np.tensordot
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3])
    print(tensor_product(a, b))
    print(np.tensordot(a, b, axes=0))
