from libc.stdint cimport uint32_t, uint64_t, UINT32_MAX
import cython
import numpy as np
from scipy import sparse
from maxflow cimport algs


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def edmonds_karp(A, source, sink):
    if type(A) is sparse.csr.csr_matrix:
        A_coo = A.tocoo(copy=False)
        n = A_coo.shape[0]
        m = A_coo.nnz
        if n <= UINT32_MAX:
            if A_coo.dtype < np.int64:
                return algs.run_ek_sparse(1, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink) # T:uint32, U:uint32
            else:
                return algs.run_ek_sparse(2, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink) # T:uint32, U:uint64
        else:
            if A_coo.dtype < np.int64:
                return algs.run_ek_sparse(3, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink) # T:uint64, U:uint32
            else:
                return algs.run_ek_sparse(4, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink) # T:uint64, U:uint64

    elif type(A) is np.ndarray:
        np.ascontiguousarray(A)
        n = A.shape[0]
        if n <= UINT32_MAX:
            if A.dtype < np.int64:
                return algs.run_ek_dense(1, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink) # T:uint32, U:uint32
            else:
                return algs.run_ek_dense(2, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink) # T:uint32, U:uint64
        else:
            if A.dtype < np.int64:
                return algs.run_ek_dense(3, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink) # T:uint64, U:uint32
            else:
                return algs.run_ek_dense(4, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink) # T:uint64, U:uint64



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def push_relabel_fifo(A, source, sink):
    if type(A) is sparse.csr.csr_matrix:
        A_coo = A.tocoo(copy=False)
        n = A_coo.shape[0]
        m = A_coo.nnz
        if n <= UINT32_MAX:
            if A_coo.dtype < np.int64:
                return algs.run_prf_sparse(1, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink) # T:uint32, U:uint32
            else:
                return algs.run_prf_sparse(2, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink) # T:uint32, U:uint64
        else:
            if A_coo.dtype < np.int64:
                return algs.run_prf_sparse(3, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink) # T:uint64, U:uint32
            else:
                return algs.run_prf_sparse(4, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink) # T:uint64, U:uint64

    elif type(A) is np.ndarray:
        np.ascontiguousarray(A)
        n = A.shape[0]
        if A.shape[0] <= UINT32_MAX:
            if A.dtype < np.int64:
                return algs.run_prf_dense(1, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink) # T:uint32, U:uint32
            else:
                return algs.run_prf_dense(2, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink) # T:uint32, U:uint64
        else:
            if A.dtype < np.int64:
                return algs.run_prf_dense(3, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink) # T:uint64, U:uint32
            else:
                return algs.run_prf_dense(4, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink) # T:uint64, U:uint64



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def push_relabel_highest(A, source, sink):
    if type(A) is sparse.csr.csr_matrix:
        A_coo = A.tocoo(copy=False)
        n = A_coo.shape[0]
        m = A_coo.nnz
        if n <= UINT32_MAX:
            if A_coo.dtype < np.int64:
                return algs.run_prh_sparse(1, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink) # T:uint32, U:uint32
            else:
                return algs.run_prh_sparse(2, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink) # T:uint32, U:uint64
        else:
            if A_coo.dtype < np.int64:
                return algs.run_prh_sparse(3, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink) # T:uint64, U:uint32
            else:
                return algs.run_prh_sparse(4, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink) # T:uint64, U:uint64

    elif type(A) is np.ndarray:
        np.ascontiguousarray(A)
        n = A.shape[0]
        if A.shape[0] <= UINT32_MAX:
            if A.dtype < np.int64:
                return algs.run_prh_dense(1, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink) # T:uint32, U:uint32
            else:
                return algs.run_prh_dense(2, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink) # T:uint32, U:uint64
        else:
            if A.dtype < np.int64:
                return algs.run_prh_dense(3, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink) # T:uint64, U:uint32
            else:
                return algs.run_prh_dense(4, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink) # T:uint64, U:uint64



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def dinic(A, source, sink):
    if type(A) is sparse.csr.csr_matrix:
        A_coo = A.tocoo(copy=False)
        n = A_coo.shape[0]
        m = A_coo.nnz
        if n <= UINT32_MAX:
            if A_coo.dtype < np.int64:
                return algs.run_din_sparse(1, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink) # T:uint32, U:uint32
            else:
                return algs.run_din_sparse(2, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink) # T:uint32, U:uint64
        else:
            if A_coo.dtype < np.int64:
                return algs.run_din_sparse(3, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink) # T:uint64, U:uint32
            else:
                return algs.run_din_sparse(4, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink) # T:uint64, U:uint64

    elif type(A) is np.ndarray:
        np.ascontiguousarray(A)
        n = A.shape[0]
        if A.shape[0] <= UINT32_MAX:
            if A.dtype < np.int64:
                return algs.run_din_dense(1, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink) # T:uint32, U:uint32
            else:
                return algs.run_din_dense(2, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink) # T:uint32, U:uint64
        else:
            if A.dtype < np.int64:
                return algs.run_din_dense(3, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink) # T:uint64, U:uint32
            else:
                return algs.run_din_dense(4, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink) # T:uint64, U:uint64


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def ahuja_orlin(A, source, sink):
    if type(A) is sparse.csr.csr_matrix:
        A_coo = A.tocoo(copy=False)
        n = A_coo.shape[0]
        m = A_coo.nnz
        if n <= UINT32_MAX:
            if A_coo.dtype < np.int64:
                return algs.run_ao_sparse(1, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink) # T:uint32, U:uint32
            else:
                return algs.run_ao_sparse(2, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink) # T:uint32, U:uint64
        else:
            if A_coo.dtype < np.int64:
                return algs.run_ao_sparse(3, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink) # T:uint64, U:uint32
            else:
                return algs.run_ao_sparse(4, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink) # T:uint64, U:uint64

    elif type(A) is np.ndarray:
        np.ascontiguousarray(A)
        n = A.shape[0]
        if A.shape[0] <= UINT32_MAX:
            if A.dtype < np.int64:
                return algs.run_ao_dense(1, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink) # T:uint32, U:uint32
            else:
                return algs.run_ao_dense(2, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink) # T:uint32, U:uint64
        else:
            if A.dtype < np.int64:
                return algs.run_ao_dense(3, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink) # T:uint64, U:uint32
            else:
                return algs.run_ao_dense(4, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink) # T:uint64, U:uint64



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def parallel_push_relabel(A, source, sink, nthreads=1):
    if type(A) is sparse.csr.csr_matrix:
        A_coo = A.tocoo(copy=False)
        n = A_coo.shape[0]
        m = A_coo.nnz
        if n <= UINT32_MAX:
            if A_coo.dtype < np.int64:
                return algs.run_ppr_sparse(1, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint32, U:uint32
            else:
                return algs.run_ppr_sparse(2, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint32, U:uint64
        else:
            if A_coo.dtype < np.int64:
                return algs.run_ppr_sparse(3, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint64, U:uint32
            else:
                return algs.run_ppr_sparse(4, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint64, U:uint64

    elif type(A) is np.ndarray:
        np.ascontiguousarray(A)
        n = A.shape[0]
        if A.shape[0] <= UINT32_MAX:
            if A.dtype < np.int64:
                return algs.run_ppr_dense(1, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint32, U:uint32
            else:
                return algs.run_ppr_dense(2, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint32, U:uint64
        else:
            if A.dtype < np.int64:
                return algs.run_ppr_dense(3, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint64, U:uint32
            else:
                return algs.run_ppr_dense(4, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint64, U:uint64



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def parallel_push_relabel_segment(A, source, sink, nthreads=1):
    if type(A) is sparse.csr.csr_matrix:
        A_coo = A.tocoo(copy=False)
        n = A_coo.shape[0]
        m = A_coo.nnz
        if n <= UINT32_MAX:
            if A_coo.dtype < np.int64:
                return algs.run_pprs_sparse(1, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint32, U:uint32
            else:
                return algs.run_pprs_sparse(2, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint32, U:uint64
        else:
            if A_coo.dtype < np.int64:
                return algs.run_pprs_sparse(3, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint64, U:uint32
            else:
                return algs.run_pprs_sparse(4, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint64, U:uint64

    elif type(A) is np.ndarray:
        np.ascontiguousarray(A)
        n = A.shape[0]
        if A.shape[0] <= UINT32_MAX:
            if A.dtype < np.int64:
                return algs.run_pprs_dense(1, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint32, U:uint32
            else:
                return algs.run_pprs_dense(2, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint32, U:uint64
        else:
            if A.dtype < np.int64:
                return algs.run_pprs_dense(3, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint64, U:uint32
            else:
                return algs.run_pprs_dense(4, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint64, U:uint64



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def parallel_AhujaOrlin_segment(A, source, sink, nthreads=1):
    if type(A) is sparse.csr.csr_matrix:
        A_coo = A.tocoo(copy=False)
        n = A_coo.shape[0]
        m = A_coo.nnz
        if n <= UINT32_MAX:
            if A_coo.dtype < np.int64:
                return algs.run_paos_sparse(1, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint32, U:uint32
            else:
                return algs.run_paos_sparse(2, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint32, U:uint64
        else:
            if A_coo.dtype < np.int64:
                return algs.run_paos_sparse(3, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint64, U:uint32
            else:
                return algs.run_paos_sparse(4, <size_t> A_coo.data.ctypes.data, <size_t> A_coo.row.ctypes.data, <size_t> A_coo.col.ctypes.data, <size_t> n, <size_t> m, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint64, U:uint64

    elif type(A) is np.ndarray:
        np.ascontiguousarray(A)
        n = A.shape[0]
        if A.shape[0] <= UINT32_MAX:
            if A.dtype < np.int64:
                return algs.run_paos_dense(1, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint32, U:uint32
            else:
                return algs.run_paos_dense(2, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint32, U:uint64
        else:
            if A.dtype < np.int64:
                return algs.run_paos_dense(3, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint64, U:uint32
            else:
                return algs.run_paos_dense(4, <size_t> A.ctypes.data, <size_t> n, <size_t> source, <size_t> sink, <size_t> nthreads) # T:uint64, U:uint64
