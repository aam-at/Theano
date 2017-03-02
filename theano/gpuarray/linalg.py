from __future__ import absolute_import, division, print_function

import pkg_resources
import warnings

from theano import Op, Apply
from theano.gpuarray import GpuArrayType
from theano.gpuarray.basic_ops import as_gpuarray_variable, infer_context_name

import numpy as np
from numpy.linalg.linalg import LinAlgError

try:
    import pygpu
except ImportError:
    pass

cusolver_available = False
try:
    import skcuda
    from skcuda import cusolver
    cusolver_available = True
except (ImportError, OSError, RuntimeError, pkg_resources.DistributionNotFound):
    pass

if cusolver_available:
    # Add cusolver call as it is missing in skcuda
    # SPOTRS
    cusolver._libcusolver.cusolverDnSpotrs.restype = int
    cusolver._libcusolver.cusolverDnSpotrs.argtypes = [cusolver.ctypes.c_void_p,
                                                       cusolver.ctypes.c_int,
                                                       cusolver.ctypes.c_int,
                                                       cusolver.ctypes.c_int,
                                                       cusolver.ctypes.c_void_p,
                                                       cusolver.ctypes.c_int,
                                                       cusolver.ctypes.c_void_p,
                                                       cusolver.ctypes.c_int,
                                                       cusolver.ctypes.c_void_p]

    def cusolverDnSpotrs(handle, uplo, n, nrhs, A, lda,
                         B, ldb, devInfo):
        """
        Solve real single precision linear system for hermitian matrices.
        References
        ----------
        `cusolverDn<t>potrs <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrs>`_
        """

        status = cusolver._libcusolver.cusolverDnSpotrs(handle, uplo, n, nrhs,
                                                        int(A), lda, int(B),
                                                        ldb, int(devInfo))
        cusolver.cusolverCheckStatus(status)


class GpuCusolverSolve(Op):
    """
    CUSOLVER GPU solver OP.

    Parameters
    ----------
    trans
        Whether to take the transpose of the input matrix or not.

    """

    __props__ = ('A_structure', 'trans', 'inplace')

    def __init__(self, A_structure='general', trans='N', inplace=False):
        self.trans = trans
        self.inplace = inplace
        self.A_structure = A_structure
        if self.inplace:
            self.destroy_map = {0: [0, 1]}
        super(GpuCusolverSolve, self).__init__()

    def make_node(self, inp1, inp2):
        if not cusolver_available:
            raise RuntimeError('CUSOLVER is not available and '
                               'GpuCusolverSolve Op can not be constructed.')
        if skcuda.__version__ <= '0.5.1':
            warnings.warn('The GpuSolve op requires scikit-cuda > 0.5.1 to work with CUDA 8')
        context_name = infer_context_name(inp1, inp2)

        inp1 = as_gpuarray_variable(inp1, context_name)
        inp2 = as_gpuarray_variable(inp2, context_name)

        inp1 = gpu_contiguous(inp1)
        inp2 = gpu_contiguous(inp2)

        # this op can only operate on float32 matrices
        assert inp1.ndim == 2
        assert inp2.ndim == 2
        assert inp1.dtype == 'float32'
        assert inp2.dtype == 'float32'

        return Apply(
            self, [inp1, inp2],
            [GpuArrayType('float32',
                          broadcastable=inp1.broadcastable,
                          context_name=context_name)()])

    def prepare_node(self, node, storage_map, compute_map, impl):
        ctx = node.inputs[0].type.context
        handle = getattr(ctx, 'cusolver_handle', None)
        if handle is None:
            with ctx:
                ctx.cusolver_handle = cusolver.cusolverDnCreate()

    def check_dev_info(self, dev_info):
        val = np.asarray(dev_info)[0]
        if val > 0:
            raise LinAlgError('A is singular')

    def perform(self, node, inputs, outputs):
        context = inputs[0][0].context

        # Size of the matrices to invert.
        z = outputs[0]

        # Matrix.
        A = inputs[0]

        # Solution vectors.
        b = inputs[1]

        assert(len(A.shape) == 2)
        assert(len(b.shape) == 2)

        if self.trans in ['T', 'C']:
            trans = 1
            l, n = A.shape
            k, m = b.shape
        elif self.trans == 'N':
            trans = 0
            n, l = A.shape
            k, m = b.shape
        else:
            raise ValueError('Invalid value for trans')
        if l != n:
            raise ValueError('A must be a square matrix')
        if n != k:
            raise ValueError('A and b must be aligned.')

        lda = max(1, n)
        ldb = max(1, k)

        # We copy A and b as cusolver operates inplace
        b = pygpu.array(b, copy=True, order='F')
        if not self.inplace:
            A = pygpu.array(A, copy=True)
        A_ptr = A.gpudata
        b_ptr = b.gpudata

        # cusolver expects a F ordered matrix, but A is not explicitly
        # converted between C and F order, instead we switch the
        # "transpose" flag.
        if A.flags['C_CONTIGUOUS']:
            trans = 1 - trans

        if self.A_structure == 'symmetric':
            with context:
                workspace_size = cusolver.cusolverDnSpotrf_bufferSize(
                    context.cusolver_handle, 0, n, A_ptr, lda)

            workspace = pygpu.zeros(workspace_size, dtype='float32',
                                    context=context)

            dev_info = pygpu.zeros((1,), dtype='int32', context=context)

            workspace_ptr = workspace.gpudata
            dev_info_ptr = dev_info.gpudata

            with context:
                cusolver.cusolverDnSpotrf(
                    context.cusolver_handle, 0, n, A_ptr, lda, workspace_ptr,
                    workspace_size, dev_info_ptr)
                self.check_dev_info(dev_info)

                cusolverDnSpotrs(
                    context.cusolver_handle, 0, n, m, A_ptr, lda,
                    b_ptr, ldb, dev_info_ptr)

        else:
            # general case for A
            with context:
                workspace_size = cusolver.cusolverDnSgetrf_bufferSize(
                    context.cusolver_handle, n, n, A_ptr, lda)

            workspace = pygpu.zeros(workspace_size, dtype='float32',
                                    context=context)

            pivots = pygpu.zeros(n, dtype='int32', context=context)

            dev_info = pygpu.zeros((1,), dtype='int32', context=context)

            workspace_ptr = workspace.gpudata
            pivots_ptr = pivots.gpudata
            dev_info_ptr = dev_info.gpudata

            with context:
                cusolver.cusolverDnSgetrf(
                    context.cusolver_handle, n, n, A_ptr, lda, workspace_ptr,
                    pivots_ptr, dev_info_ptr)
                self.check_dev_info(dev_info)

                cusolver.cusolverDnSgetrs(
                    context.cusolver_handle, trans, n, m, A_ptr, lda,
                    pivots_ptr, b_ptr, ldb, dev_info_ptr)

        z[0] = b


def gpu_solve(A, b, A_structure='general', trans='N'):
    return GpuCusolverSolve(A_structure, trans)(A, b)


class GpuCusolverSVD(Op):
    __props__ = ('full_matrices', 'compute_uv')

    def __init__(self, full_matrices=True, compute_uv=True):
        self.full_matrices = full_matrices
        self.compute_uv = compute_uv

    def make_node(self, A):
        if not cusolver_available:
            raise RuntimeError('CUSOLVER is not available and '
                               'GpuCusolverSolve Op can not be constructed.')
        if not self.full_matrices or not self.compute_uv:
            raise ValueError("CUSOLVER only supports jobu = jobvt = 'A'")

        context_name = infer_context_name(A)
        A = as_gpuarray_variable(A, context_name)
        assert A.ndim == 2, "The input of svd function should be a matrix."
        assert A.dtype == 'float32'

        return Apply(self, [A], [A.type(),
                                 GpuArrayType(A.dtype, broadcastable=[False],
                                              context_name=context_name)(),
                                 A.type()])

    def prepare_node(self, node, storage_map, compute_map, impl):
        ctx = node.inputs[0].type.context
        handle = getattr(ctx, 'cusolver_handle', None)
        if handle is None:
            with ctx:
                ctx.cusolver_handle = cusolver.cusolverDnCreate()

    def perform(self, node, inputs, outputs):
        ctx = inputs[0][0].context
        x = inputs[0]
        data_type = x.dtype
        assert len(x.shape) == 2
        m, n = x.shape
        u, s, v = outputs

        # allocate output arrays
        s_gpu = pygpu.empty(min(m, n), dtype=data_type, context=ctx)
        u_gpu = pygpu.empty((m, m), dtype=data_type, context=ctx)
        vh_gpu = pygpu.empty((n, n), dtype=data_type, context=ctx)

        with ctx:
            workspace_size = cusolver.cusolverDnSgesvd_bufferSize(
                ctx.cusolver_handle, m, n)
        workspace = pygpu.zeros(workspace_size, dtype=x.dtype, context=ctx)
        dev_info = pygpu.zeros((1,), dtype='int32', context=ctx)

        # call cusolverDnSgesvd using scikit-cuda
        x_ptr = x.gpudata
        s_ptr = s_gpu.gpudata
        u_ptr = u_gpu.gpudata
        vh_ptr = vh_gpu.gpudata
        workspace_ptr = workspace.gpudata
        dev_info_ptr = dev_info.gpudata

        with ctx:
            cusolver.cusolverDnSgesvd(ctx.cusolver_handle, 'A', 'A', m, n,
                                      x_ptr, m, s_ptr, u_ptr, m, vh_ptr, n,
                                      workspace_ptr, workspace_size, 0, dev_info_ptr)
            if np.asarray(dev_info[0]) > 0:
                raise LinAlgError('SVD did not converge')

        u[0], s[0], v[0] = u_gpu, s_gpu, vh_gpu


def gpu_svd(A, full_matrices=1, compute_uv=1):
    """
    This function performs the SVD on CPU.

    Parameters
    ----------
    full_matrices : bool, optional
        If True (default), u and v have the shapes (M, M) and (N, N),
        respectively.
        Otherwise, the shapes are (M, K) and (K, N), respectively,
        where K = min(M, N).
    compute_uv : bool, optional
        Whether or not to compute u and v in addition to s.
        True by default.

    Returns
    -------
    U, S, V : matrices

    """
    return GpuCusolverSVD(full_matrices, compute_uv)(A)
