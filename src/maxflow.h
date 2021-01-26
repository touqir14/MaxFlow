#include <ios>
#include <fstream>
#include "lib/common_types.h"
#include <vector>
#include "graph_loader.h"
#include "lib/algorithms/parallel/push_relabel_segment.h"
#include "lib/algorithms/sequential/ahuja_orlin.h"
#include "lib/algorithms/parallel/parallel_push_relabel.h"
#include "lib/algorithms/sequential/push_relabel_highest.h"
#include "lib/algorithms/sequential/push_relabel_fifo.h"
#include "lib/algorithms/sequential/edmonds_karp.h"
#include "lib/algorithms/sequential/dinic.h"
#include "lib/algorithms/parallel/ahuja_orlin_segment.h"

#define DEFAULT_VECTOR std::vector

template <template <typename, typename> typename EDGE, template <template <typename> typename, typename, typename> typename alg, template <class> typename vector = std::vector>
std::size_t _run_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, size_t nthreads=1) {
    switch(mode) {
        case 1: 
            {
                auto graph = load_graph_dense<uint32_t, uint32_t, EDGE> ((void*) A_ptr, n);
                alg<vector, uint32_t, uint32_t> M(std::move(graph), source, sink, nthreads);                
                return M.find_max_flow();
            }
        case 2:
            {
                auto graph = load_graph_dense<uint32_t, uint64_t, EDGE> ((void*) A_ptr, n);
                alg<vector, uint32_t, uint64_t> M(std::move(graph), source, sink, nthreads);
                return M.find_max_flow();
            }
        case 3:
            {
                auto graph = load_graph_dense<uint64_t, uint32_t, EDGE> ((void*) A_ptr, n);
                alg<vector, uint64_t, uint32_t> M(std::move(graph), source, sink, nthreads);
                return M.find_max_flow();
            }
        case 4:
            {
                auto graph = load_graph_dense<uint64_t, uint64_t, EDGE> ((void*) A_ptr, n);
                alg<vector, uint64_t, uint64_t> M(std::move(graph), source, sink, nthreads);
                return M.find_max_flow();
            }
    }
}


std::size_t run_ek_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink) {
    return _run_dense<basic_edge, edmonds_karp::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, n, source, sink);
}

std::size_t run_prf_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink) {
    return _run_dense<cached_edge, push_relabel_fifo::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, n, source, sink);
}

std::size_t run_prh_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink) {
    return _run_dense<cached_edge, push_relabel_highest::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, n, source, sink);
}

std::size_t run_ao_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink) {
    return _run_dense<cached_edge, ahuja_orlin::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, n, source, sink);
}

std::size_t run_din_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink) {
    return _run_dense<basic_edge, dinic::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, n, source, sink);
}

std::size_t run_ppr_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, size_t nthreads) {
    return _run_dense<cached_edge, parallel_push_relabel::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, n, source, sink, nthreads);
}

std::size_t run_pprs_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, size_t nthreads) {
    return _run_dense<cached_edge, push_relabel_segment::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, n, source, sink, nthreads);
}

std::size_t run_paos_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, size_t nthreads) {
    return _run_dense<cached_edge, ahuja_orlin_segment::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, n, source, sink, nthreads);
}


template <template <typename, typename> typename EDGE, template <template <typename> typename, typename, typename> typename alg, template <class> typename vector = std::vector>
std::size_t _run_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, size_t nthreads=1) {
    switch(mode) {
        case 1: 
            {
                auto graph = load_graph_sparse<uint32_t, uint32_t, EDGE> ((void*) A_ptr, (void*) row_ptr, (void*) col_ptr, n, m);
                alg<vector, uint32_t, uint32_t> M(std::move(graph), source, sink, nthreads);                
                return M.find_max_flow();
            }
        case 2:
            {
                auto graph = load_graph_sparse<uint32_t, uint64_t, EDGE> ((void*) A_ptr, (void*) row_ptr, (void*) col_ptr, n, m);
                alg<vector, uint32_t, uint64_t> M(std::move(graph), source, sink, nthreads);
                return M.find_max_flow();
            }
        case 3:
            {
                auto graph = load_graph_sparse<uint64_t, uint32_t, EDGE> ((void*) A_ptr, (void*) row_ptr, (void*) col_ptr, n, m);
                alg<vector, uint64_t, uint32_t> M(std::move(graph), source, sink, nthreads);
                return M.find_max_flow();
            }
        case 4:
            {
                auto graph = load_graph_sparse<uint64_t, uint64_t, EDGE> ((void*) A_ptr, (void*) row_ptr, (void*) col_ptr, n, m);
                alg<vector, uint64_t, uint64_t> M(std::move(graph), source, sink, nthreads);
                return M.find_max_flow();
            }
    }
}


std::size_t run_ek_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink) {
    return _run_sparse<basic_edge, edmonds_karp::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink);
}

std::size_t run_prf_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink) {
    return _run_sparse<cached_edge, push_relabel_fifo::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink);
}

std::size_t run_prh_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink) {
    return _run_sparse<cached_edge, push_relabel_highest::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink);
}

std::size_t run_ao_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink) {
    return _run_sparse<cached_edge, ahuja_orlin::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink);
}

std::size_t run_din_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink) {
    return _run_sparse<basic_edge, dinic::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink);
}

std::size_t run_ppr_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, size_t nthreads) {
    return _run_sparse<cached_edge, parallel_push_relabel::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink, nthreads);
}

std::size_t run_pprs_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, size_t nthreads) {
    return _run_sparse<cached_edge, push_relabel_segment::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink, nthreads);
}

std::size_t run_paos_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, size_t nthreads) {
    return _run_sparse<cached_edge, ahuja_orlin_segment::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink, nthreads);
}
