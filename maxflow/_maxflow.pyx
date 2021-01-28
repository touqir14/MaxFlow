from libc.stdint cimport uint32_t, uint64_t, UINT32_MAX, int32_t
import cython
from maxflow cimport algs

cdef int __graph_index_next=-1


cpdef int _destroy_graph(int graph_idx):
    return algs.destroy_graph(graph_idx)


cpdef (size_t, int) _edmonds_karp_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx):
    global __graph_index_next
    cdef size_t max_flow = algs.run_ek_dense(mode, A_ptr, n, source, sink, graph_idx)
    if graph_idx == -2 or graph_idx == -3:
        __graph_index_next += 1
    return max_flow, __graph_index_next


cpdef (size_t, int) _edmonds_karp_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx):
    global __graph_index_next
    cdef size_t max_flow = algs.run_ek_sparse(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink, graph_idx)
    if graph_idx == -2 or graph_idx == -3:
        __graph_index_next += 1
    return max_flow, __graph_index_next


cpdef (size_t, int) _push_relabel_fifo_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx):
    global __graph_index_next
    cdef size_t max_flow = algs.run_prf_dense(mode, A_ptr, n, source, sink, graph_idx)
    if graph_idx == -2 or graph_idx == -3:
        __graph_index_next += 1
    return max_flow, __graph_index_next


cpdef (size_t, int) _push_relabel_fifo_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx):
    global __graph_index_next
    cdef size_t max_flow = algs.run_prf_sparse(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink, graph_idx)
    if graph_idx == -2 or graph_idx == -3:
        __graph_index_next += 1
    return max_flow, __graph_index_next


cpdef (size_t, int) _push_relabel_highest_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx):
    global __graph_index_next
    cdef size_t max_flow = algs.run_prh_dense(mode, A_ptr, n, source, sink, graph_idx)
    if graph_idx == -2 or graph_idx == -3:
        __graph_index_next += 1
    return max_flow, __graph_index_next


cpdef (size_t, int) _push_relabel_highest_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx):
    global __graph_index_next
    cdef size_t max_flow = algs.run_prh_sparse(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink, graph_idx)
    if graph_idx == -2 or graph_idx == -3:
        __graph_index_next += 1
    return max_flow, __graph_index_next


cpdef (size_t, int) _dinic_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx):
    global __graph_index_next
    cdef size_t max_flow = algs.run_din_dense(mode, A_ptr, n, source, sink, graph_idx)
    if graph_idx == -2 or graph_idx == -3:
        __graph_index_next += 1
    return max_flow, __graph_index_next


cpdef (size_t, int) _dinic_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx):
    global __graph_index_next
    cdef size_t max_flow = algs.run_din_sparse(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink, graph_idx)
    if graph_idx == -2 or graph_idx == -3:
        __graph_index_next += 1
    return max_flow, __graph_index_next


cpdef (size_t, int) _ahuja_orlin_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx):
    global __graph_index_next
    cdef size_t max_flow = algs.run_ao_dense(mode, A_ptr, n, source, sink, graph_idx)
    if graph_idx == -2 or graph_idx == -3:
        __graph_index_next += 1
    return max_flow, __graph_index_next


cpdef (size_t, int) _ahuja_orlin_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx):
    global __graph_index_next
    cdef size_t max_flow = algs.run_ao_sparse(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink, graph_idx)
    if graph_idx == -2 or graph_idx == -3:
        __graph_index_next += 1
    return max_flow, __graph_index_next


cpdef (size_t, int) _parallel_push_relabel_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx, int nthreads=1):
    global __graph_index_next
    cdef size_t max_flow = algs.run_ppr_dense(mode, A_ptr, n, source, sink, graph_idx, nthreads)
    if graph_idx == -2 or graph_idx == -3:
        __graph_index_next += 1
    return max_flow, __graph_index_next


cpdef (size_t, int) _parallel_push_relabel_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx, int nthreads=1):
    global __graph_index_next
    cdef size_t max_flow = algs.run_ppr_sparse(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink, graph_idx, nthreads)
    if graph_idx == -2 or graph_idx == -3:
        __graph_index_next += 1
    return max_flow, __graph_index_next


cpdef (size_t, int) _parallel_push_relabel_segment_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx, int nthreads=1):
    global __graph_index_next
    cdef size_t max_flow = algs.run_pprs_dense(mode, A_ptr, n, source, sink, graph_idx, nthreads)
    if graph_idx == -2 or graph_idx == -3:
        __graph_index_next += 1
    return max_flow, __graph_index_next


cpdef (size_t, int) _parallel_push_relabel_segment_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx, int nthreads=1):
    global __graph_index_next
    cdef size_t max_flow = algs.run_pprs_sparse(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink, graph_idx, nthreads)
    if graph_idx == -2 or graph_idx == -3:
        __graph_index_next += 1
    return max_flow, __graph_index_next


cpdef (size_t, int) _parallel_AhujaOrlin_segment_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx, int nthreads=1):
    global __graph_index_next
    cdef size_t max_flow = algs.run_paos_dense(mode, A_ptr, n, source, sink, graph_idx, nthreads)
    if graph_idx == -2 or graph_idx == -3:
        __graph_index_next += 1
    return max_flow, __graph_index_next


cpdef (size_t, int) _parallel_AhujaOrlin_segment_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx, int nthreads=1):
    global __graph_index_next
    cdef size_t max_flow = algs.run_paos_sparse(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink, graph_idx, nthreads)
    if graph_idx == -2 or graph_idx == -3:
        __graph_index_next += 1
    return max_flow, __graph_index_next










