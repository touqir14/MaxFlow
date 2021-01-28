#ifndef MAXFLOW_GRAPH_LOADER_H
#define MAXFLOW_GRAPH_LOADER_H

#include <fstream>
#include <sstream>
#include <iterator>
#include <iostream>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cassert>
#include "lib/common_types.h"
#include <queue>
#include <algorithm>
#include <vector>
#include <type_traits>


template <typename T, typename U, template <typename, typename> typename EDGE = basic_edge>
void _init_graph(std::size_t num_edges, std::vector<T> &outgoing_edge_cnt, std::vector<EDGE<T, U>> &edges, std::vector<std::vector<EDGE<T, U>>> &graph)
{
    //alloc graph
    for (std::size_t i = 0; i < graph.size(); ++i) 
    {
        graph[i].reserve(outgoing_edge_cnt[i]);
    }

    //insert forward edges
    for (std::size_t i = 0; i < num_edges; i += 2)
    {
        auto &edge = edges[i];
        auto &reverse_edge = edges[i + 1];
        graph[reverse_edge.dst_vertex].emplace_back(edge.dst_vertex, edge.r_capacity, i + 1);
    }

    auto sizes = std::make_unique<T[]> (graph.size());
    for (std::size_t i = 0; i < graph.size(); ++i)
        sizes[i] = graph[i].size();

    //insert backward edges
    for (std::size_t i = 0; i < graph.size(); ++i)
    {
        for (std::size_t k = 0; k < sizes[i]; ++k)
        {
            auto &edge = graph[i][k];
            auto reverse_edge = edges[edge.reverse_edge_index]; 
            edge.reverse_edge_index = graph[edge.dst_vertex].size();
            graph[edge.dst_vertex].emplace_back(i, reverse_edge.r_capacity, k);
        }
    }
}


// Set reverse_r_capacity for cached edges used in push-relabel methods.
template <typename T, typename U>
void set_reverse_edge_cap(std::vector<std::vector<cached_edge<T, U>>> & graph)
{
    for (auto &vec : graph)
        for (auto &edge : vec)
            edge.reverse_r_capacity = graph[edge.dst_vertex][edge.reverse_edge_index].r_capacity;
}


template<typename T, typename U, template <typename, typename> typename EDGE>
auto _load_graph_dense(void* A_ptr, size_t n) {
    
    const U* capacity_array = (U*) A_ptr;
    std::vector<EDGE<T,U>> edges;
    std::vector<T> outgoing_edge_cnt(n);
    std::size_t num_edges = 0;
    edges.resize(n*n);

    for (std::size_t i=0; i<n; ++i) {
        for (std::size_t j=i+1; j<n; ++j) {
            std::size_t forward_idx = i*n + j;
            std::size_t reverse_idx = j*n + i;
            if (capacity_array[forward_idx] != 0) {
                edges[num_edges].dst_vertex = j;
                edges[num_edges].r_capacity = capacity_array[forward_idx];
                edges[num_edges+1].dst_vertex = i;
                edges[num_edges+1].r_capacity = capacity_array[reverse_idx];
                outgoing_edge_cnt[i] += 1;
                outgoing_edge_cnt[j] += 1;
                num_edges += 2;
            }
            else if (capacity_array[reverse_idx] != 0) {
                edges[num_edges].dst_vertex = i;
                edges[num_edges].r_capacity = capacity_array[reverse_idx];
                edges[num_edges+1].dst_vertex = j;
                edges[num_edges+1].r_capacity = 0;
                outgoing_edge_cnt[i] += 1;
                outgoing_edge_cnt[j] += 1;
                num_edges += 2;
            }
        }
    }

    edges.resize(num_edges);
    auto graph_ptr = std::make_shared<std::vector<std::vector<EDGE<T, U>>>> (n);
    _init_graph<T, U, EDGE> (num_edges, outgoing_edge_cnt, edges, *graph_ptr);
    if constexpr(std::is_same_v<EDGE<T,U>, cached_edge<T,U>>) {
        set_reverse_edge_cap<T, U>(*graph_ptr);
    }
    return graph_ptr;
}



template<typename T, typename U, template <typename, typename> typename EDGE>
auto _load_graph_sparse(void* A_ptr, void* row_ptr, void* col_ptr, size_t n, size_t m) {
    
    const U* capacity_array = (U*) A_ptr;
    const T* rows = (T*) row_ptr;
    const T* cols = (T*) col_ptr;
    std::vector<EDGE<T,U>> edges;
    std::vector<T> outgoing_edge_cnt(n);
    std::vector<std::unordered_map<T, T>> edge_map(n);
    size_t num_edges = 0;
    T src, dst;
    U cap;
    auto undefined = std::numeric_limits<T>::max ();

    if ((2*m) > (n*n)) {
        edges.reserve(n*n);
    }
    else {
        edges.reserve(2*m);
    }

    for (size_t i=0; i<m; ++i) {
        cap = capacity_array[i];
        src = rows[i];
        dst = cols[i];
        auto it = edge_map[dst].find(src);
        if (it != std::end(edge_map[dst])){
            edges[it->second+1].r_capacity += cap;
        }
        else {
            edges.emplace_back(dst, cap, undefined);
            edges.emplace_back(src, 0, undefined);
            edge_map[src].emplace(dst, num_edges);
            outgoing_edge_cnt[src] += 1;
            outgoing_edge_cnt[dst] += 1;
            num_edges += 2;
        }
    }

    edges.shrink_to_fit();
    auto graph_ptr = std::make_shared<std::vector<std::vector<EDGE<T, U>>>> (n);
    _init_graph<T, U, EDGE> (num_edges, outgoing_edge_cnt, edges, *graph_ptr);
    if constexpr(std::is_same_v<EDGE<T,U>, cached_edge<T,U>>) {
        set_reverse_edge_cap<T, U>(*graph_ptr);
    }
    return graph_ptr;
}

    
#endif //MAXFLOW_GRAPH_LOADER_H
