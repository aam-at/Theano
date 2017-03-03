from __future__ import absolute_import, division, print_function

import unittest
import numpy as np
import theano

from theano.tests import unittest_tools as utt
from .config import mode_with_gpu, mode_without_gpu
from .test_basic_ops import rand


from numpy.linalg.linalg import LinAlgError

# Skip tests if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
from theano.gpuarray.linalg import (cusolver_available, gpu_solve, GpuCusolverSVD)
from theano.tensor.nlinalg import SVD

if not cusolver_available:
    raise SkipTest('Optional package scikits.cuda.cusolver not available')


class TestCusolver(unittest.TestCase):
    def run_gpu_solve(self, A_val, x_val, A_struct=None):
        b_val = np.dot(A_val, x_val)
        b_val_trans = np.dot(A_val.T, x_val)

        A = theano.tensor.matrix("A", dtype="float32")
        b = theano.tensor.matrix("b", dtype="float32")
        b_trans = theano.tensor.matrix("b", dtype="float32")

        if A_struct is None:
            solver = gpu_solve(A, b)
            solver_trans = gpu_solve(A, b_trans, trans='T')
        else:
            solver = gpu_solve(A, b, A_struct)
            solver_trans = gpu_solve(A, b_trans, A_struct, trans='T')

        fn = theano.function([A, b, b_trans], [solver, solver_trans], mode=mode_with_gpu)
        res = fn(A_val, b_val, b_val_trans)
        x_res = np.array(res[0])
        x_res_trans = np.array(res[1])
        utt.assert_allclose(x_val, x_res)
        utt.assert_allclose(x_val, x_res_trans)

    def test_diag_solve(self):
        np.random.seed(1)
        A_val = np.asarray([[2, 0, 0], [0, 1, 0], [0, 0, 1]],
                           dtype="float32")
        x_val = np.random.uniform(-0.4, 0.4, (A_val.shape[1],
                                  1)).astype("float32")
        self.run_gpu_solve(A_val, x_val)

    def test_bshape_solve(self):
        """
        Test when shape of b (k, m) is such as m > k
        """
        np.random.seed(1)
        A_val = np.asarray([[2, 0, 0], [0, 1, 0], [0, 0, 1]],
                           dtype="float32")
        x_val = np.random.uniform(-0.4, 0.4, (A_val.shape[1],
                                  A_val.shape[1] + 1)).astype("float32")
        self.run_gpu_solve(A_val, x_val)

    def test_sym_solve(self):
        np.random.seed(1)
        A_val = np.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        A_sym = np.dot(A_val, A_val.T)
        x_val = np.random.uniform(-0.4, 0.4, (A_val.shape[1],
                                  1)).astype("float32")
        self.run_gpu_solve(A_sym, x_val, 'symmetric')

    def test_orth_solve(self):
        np.random.seed(1)
        A_val = np.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        A_orth = np.linalg.svd(A_val)[0]
        x_val = np.random.uniform(-0.4, 0.4, (A_orth.shape[1],
                                  1)).astype("float32")
        self.run_gpu_solve(A_orth, x_val)

    def test_uni_rand_solve(self):
        np.random.seed(1)
        A_val = np.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        x_val = np.random.uniform(-0.4, 0.4,
                                  (A_val.shape[1], 4)).astype("float32")
        self.run_gpu_solve(A_val, x_val)

    def test_linalgerrsym_solve(self):
        np.random.seed(1)
        A_val = np.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        x_val = np.random.uniform(-0.4, 0.4,
                                  (A_val.shape[1], 4)).astype("float32")
        A_val = np.dot(A_val.T, A_val)
        # make A singular
        A_val[:, 2] = A_val[:, 1] + A_val[:, 3]

        A = theano.tensor.matrix("A", dtype="float32")
        b = theano.tensor.matrix("b", dtype="float32")
        solver = gpu_solve(A, b, 'symmetric')

        fn = theano.function([A, b], [solver], mode=mode_with_gpu)
        self.assertRaises(LinAlgError, fn, A_val, x_val)

    def test_linalgerr_solve(self):
        np.random.seed(1)
        A_val = np.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        x_val = np.random.uniform(-0.4, 0.4,
                                  (A_val.shape[1], 4)).astype("float32")
        # make A singular
        A_val[:, 2] = 0

        A = theano.tensor.matrix("A", dtype="float32")
        b = theano.tensor.matrix("b", dtype="float32")
        solver = gpu_solve(A, b, trans='T')

        fn = theano.function([A, b], [solver], mode=mode_with_gpu)
        self.assertRaises(LinAlgError, fn, A_val, x_val)

    def test_linalg_svd(self):
        test_shapes = [(5, 5), (5, 7), (10, 10), (10, 15), (20, 20),
                       (20, 30), (50, 50), (50, 75)]

        svd_op = SVD()
        for shp in test_shapes:
            A_v = rand(*shp)
            A = theano.shared(A_v, 'A')
            f = theano.function([], svd_op(A), mode=mode_without_gpu)
            f2 = theano.function([], svd_op(A), mode=mode_with_gpu)
            assert any([isinstance(node.op, SVD)
                        for node in f.maker.fgraph.toposort()])
            assert any([isinstance(node.op, GpuCusolverSVD)
                        for node in f2.maker.fgraph.toposort()])
            u, s, vh = f()
            u2, s2, vh2 = f2()
            # svd is ambiguous with signs
            assert np.allclose(np.abs(u), np.abs(u2), atol=1e-3)
            assert np.allclose(np.abs(vh), np.abs(vh2), atol=1e-3)
            assert np.allclose(s, s2, atol=1e-3)

    def test_linalgerr_svd(self):
        with self.assertRaises(ValueError):
            GpuCusolverSVD(full_matrices=False, compute_uv=True)
        with self.assertRaises(ValueError):
            GpuCusolverSVD(full_matrices=True, compute_uv=False)
        with self.assertRaises(LinAlgError):
            A = theano.shared(rand(3, 2), 'A')
            f = theano.function([], SVD()(A), mode=mode_with_gpu)
            f()
