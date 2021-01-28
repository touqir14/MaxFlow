from libc.stdint cimport uint32_t, uint64_t

cdef extern from "../src/maxflow.h":
    size_t run_ek_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx)
    size_t run_prf_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx)
    size_t run_prh_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx)
    size_t run_din_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx)
    size_t run_ao_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx)
    size_t run_ppr_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx, size_t nthreads)
    size_t run_pprs_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx, size_t nthreads)
    size_t run_paos_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx, size_t nthreads)

    size_t run_ek_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx)
    size_t run_prf_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx)
    size_t run_prh_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx)
    size_t run_din_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx)
    size_t run_ao_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx)
    size_t run_ppr_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx, size_t nthreads)
    size_t run_pprs_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx, size_t nthreads)
    size_t run_paos_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx, size_t nthreads)

    size_t destroy_graph(int graph_idx)

