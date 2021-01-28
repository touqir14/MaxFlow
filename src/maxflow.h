#include <ios>
#include <iostream>
#include <fstream>
#include "lib/common_types.h"
#include <vector>
#include <unordered_map>
#include "graph_loader.h"
#include "lib/algorithms/parallel/push_relabel_segment.h"
#include "lib/algorithms/sequential/ahuja_orlin.h"
#include "lib/algorithms/parallel/parallel_push_relabel.h"
#include "lib/algorithms/sequential/push_relabel_highest.h"
#include "lib/algorithms/sequential/push_relabel_fifo.h"
#include "lib/algorithms/sequential/edmonds_karp.h"
#include "lib/algorithms/sequential/dinic.h"
#include "lib/algorithms/parallel/ahuja_orlin_segment.h"
#include "chrono"

#define DEFAULT_VECTOR std::vector

size_t g_next_idx = 0;
std::unordered_map<int, std::shared_ptr<void>> GraphMap; 

template<typename T, typename U, template <typename, typename> typename EDGE>
auto load_graph_dense(size_t A_ptr, size_t n, int graph_idx, bool& run_maxflow) {
    // Use graph_idx >= 0 for loading an existing graph and running max_flow
    // Use graph_idx = -1 for loading and running max_flow
    // Use graph_idx = -2 for loading and saving the graph
    // Use graph_idx = -3 for loading, saving the graph, and running max_flow

    run_maxflow = true;
    if (graph_idx < 0) { 
        auto graph = _load_graph_dense<T, U, EDGE>((void*) A_ptr, n); // Returns a pointer.
        if (graph_idx == -2) {
            run_maxflow = false;
            GraphMap[g_next_idx] = std::static_pointer_cast<void> (graph);
            ++g_next_idx;
        } else if (graph_idx == -3) {
            GraphMap[g_next_idx] = std::static_pointer_cast<void> (graph);
            ++g_next_idx;
        }
        return graph;
    } else {
        return std::static_pointer_cast<std::vector<std::vector<EDGE<T, U>>>> (GraphMap[graph_idx]);
    }
}


template<typename T, typename U, template <typename, typename> typename EDGE>
auto load_graph_sparse(size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, int graph_idx, bool& run_maxflow) {
    // Use graph_idx >= 0 for loading an existing graph and running max_flow
    // Use graph_idx = -1 for loading and running max_flow
    // Use graph_idx = -2 for loading and saving the graph
    // Use graph_idx = -3 for loading, saving the graph, and running max_flow

    run_maxflow = true;
    if (graph_idx < 0) { 
        auto graph = _load_graph_sparse<T, U, EDGE>((void*) A_ptr, (void*) row_ptr, (void*) col_ptr, n, m);
        if (graph_idx == -2) {
            run_maxflow = false;
            GraphMap[g_next_idx] = std::static_pointer_cast<void> (graph);
            ++g_next_idx;
        } else if (graph_idx == -3) {
            GraphMap[g_next_idx] = std::static_pointer_cast<void> (graph);
            ++g_next_idx;
        }
        return graph;
    } else {
        return std::static_pointer_cast<std::vector<std::vector<EDGE<T, U>>>> (GraphMap[graph_idx]);
    }
}

size_t destroy_graph(int graph_idx) {
    return GraphMap.erase(graph_idx);
}


template <template <typename, typename> typename EDGE, template <template <typename> typename, typename, typename> typename alg, template <class> typename vector = std::vector>
size_t _run_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx, size_t nthreads=1) {
    bool run_maxflow;
    size_t flow_value=0;
    switch(mode) {
        case 1: 
            {
                auto graph = load_graph_dense<uint32_t, uint32_t, EDGE> (A_ptr, n, graph_idx, run_maxflow);
                if (run_maxflow) {
                    alg<vector, uint32_t, uint32_t> M(*graph, source, sink, nthreads);                
                    flow_value = M.find_max_flow();
                }
                break;
            }
        case 2:
            {
                auto graph = load_graph_dense<uint32_t, uint64_t, EDGE> (A_ptr, n, graph_idx, run_maxflow);
                if (run_maxflow) {
                    alg<vector, uint32_t, uint64_t> M(*graph, source, sink, nthreads);
                    return M.find_max_flow();
                }
                break;
            }
        case 3:
            {
                auto graph = load_graph_dense<uint64_t, uint32_t, EDGE> (A_ptr, n, graph_idx, run_maxflow);
                if (run_maxflow) {
                    alg<vector, uint64_t, uint32_t> M(*graph, source, sink, nthreads);
                    return M.find_max_flow();
                }
                break;
            }
        case 4:
            {
                auto graph = load_graph_dense<uint64_t, uint64_t, EDGE> (A_ptr, n, graph_idx, run_maxflow);
                if (run_maxflow) {
                    alg<vector, uint64_t, uint64_t> M(*graph, source, sink, nthreads);
                    return M.find_max_flow();
                }
            }
    }
    return flow_value;
}


std::size_t run_ek_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx) {
    return _run_dense<basic_edge, edmonds_karp::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, n, source, sink, graph_idx);
}

std::size_t run_prf_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx) {
    return _run_dense<cached_edge, push_relabel_fifo::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, n, source, sink, graph_idx);
}

std::size_t run_prh_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx) {
    return _run_dense<cached_edge, push_relabel_highest::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, n, source, sink, graph_idx);
}

std::size_t run_ao_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx) {
    return _run_dense<cached_edge, ahuja_orlin::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, n, source, sink, graph_idx);
}

std::size_t run_din_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx) {
    return _run_dense<basic_edge, dinic::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, n, source, sink, graph_idx);
}

std::size_t run_ppr_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx, size_t nthreads) {
    return _run_dense<cached_edge, parallel_push_relabel::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, n, source, sink, graph_idx, nthreads);
}

std::size_t run_pprs_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx, size_t nthreads) {
    return _run_dense<cached_edge, push_relabel_segment::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, n, source, sink, graph_idx, nthreads);
}

std::size_t run_paos_dense(int mode, size_t A_ptr, size_t n, size_t source, size_t sink, int graph_idx, size_t nthreads) {
    return _run_dense<cached_edge, ahuja_orlin_segment::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, n, source, sink, graph_idx, nthreads);
}


template <template <typename, typename> typename EDGE, template <template <typename> typename, typename, typename> typename alg, template <class> typename vector = std::vector>
std::size_t _run_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx, size_t nthreads=1) {
    bool run_maxflow;
    switch(mode) {
        case 1: 
            {
                auto graph = load_graph_sparse<uint32_t, uint32_t, EDGE> (A_ptr, row_ptr, col_ptr, n, m, graph_idx, run_maxflow);
                if (run_maxflow) {
                    alg<vector, uint32_t, uint32_t> M(*graph, source, sink, nthreads);                
                    return M.find_max_flow();
                }
                break;
            }
        case 2:
            {
                auto graph = load_graph_sparse<uint32_t, uint64_t, EDGE> (A_ptr, row_ptr, col_ptr, n, m, graph_idx, run_maxflow);
                if (run_maxflow) {
                    alg<vector, uint32_t, uint64_t> M(*graph, source, sink, nthreads);
                    return M.find_max_flow();
                }
                break;
            }
        case 3:
            {
                auto graph = load_graph_sparse<uint64_t, uint32_t, EDGE> (A_ptr, row_ptr, col_ptr, n, m, graph_idx, run_maxflow);
                if (run_maxflow) {
                    alg<vector, uint64_t, uint32_t> M(*graph, source, sink, nthreads);
                    return M.find_max_flow();
                }
                break;
            }
        case 4:
            {
                auto graph = load_graph_sparse<uint64_t, uint64_t, EDGE> (A_ptr, row_ptr, col_ptr, n, m, graph_idx, run_maxflow);
                if (run_maxflow) {
                    alg<vector, uint64_t, uint64_t> M(*graph, source, sink, nthreads);
                    return M.find_max_flow();
                }
                break;
            }
    }
    return 0;
}


std::size_t run_ek_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx) {
    return _run_sparse<basic_edge, edmonds_karp::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink, graph_idx);
}

std::size_t run_prf_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx) {
    return _run_sparse<cached_edge, push_relabel_fifo::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink, graph_idx);
}

std::size_t run_prh_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx) {
    return _run_sparse<cached_edge, push_relabel_highest::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink, graph_idx);
}

std::size_t run_ao_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx) {
    return _run_sparse<cached_edge, ahuja_orlin::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink, graph_idx);
}

std::size_t run_din_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx) {
    return _run_sparse<basic_edge, dinic::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink, graph_idx);
}

std::size_t run_ppr_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx, size_t nthreads) {
    return _run_sparse<cached_edge, parallel_push_relabel::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink, graph_idx, nthreads);
}

std::size_t run_pprs_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx, size_t nthreads) {
    return _run_sparse<cached_edge, push_relabel_segment::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink, graph_idx, nthreads);
}

std::size_t run_paos_sparse(int mode, size_t A_ptr, size_t row_ptr, size_t col_ptr, size_t n, size_t m, size_t source, size_t sink, int graph_idx, size_t nthreads) {
    return _run_sparse<cached_edge, ahuja_orlin_segment::max_flow_instance, DEFAULT_VECTOR>(mode, A_ptr, row_ptr, col_ptr, n, m, source, sink, graph_idx, nthreads);
}
