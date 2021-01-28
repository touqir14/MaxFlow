# MaxFlow
A high performance Python library for computing the maximum flow in Graphs! This is based on the work of Jan Groschaft.
It currently contains the following algorithms:
* Edmonds-Karp's Algorithm : ```maxflow.edmonds_karp(A, source, sink)```
* Ahuja-Orlin's Algorithm : ```maxflow.ahuja_orlin(A, source, sink)```
* Dinic's Algorithm : ```maxflow.dinic(A, source, sink)```
* Push-relabel algorithm with FIFO vertex selection Algorithm : ```maxflow.push_relabel_fifo(A, source, sink)```
* Push-relabel algorithm with highest label vertex selection Algorithm: ```maxflow.push_relabel_highest(A, source, sink)```
* Parallel push-relabel Algorithm : ```maxflow.parallel_push_relabel(A, source, sink, nthreads)```
* Parallel push-relabel segment Algorithm : ```maxflow.parallel_push_relabel_segment(A, source, sink, nthreads)```
* Parallel Ahuja-Orlin segment algorithm : ```maxflow.parallel_AhujaOrlin_segment(A, source, sink, nthreads)```

## Installation
Along with Cython, a C++17 compatible compiler such as g++ >= 8 or clang++ >= 8 is required for building the extensions. Clone the repository and run ```python3 setup.py install``` from within ```MaxFlow``` directory. OpenMP is also required for the parallel algorithms.

## Usage
All the functions require a ```n x n``` Numpy array or ```scipy.sparse.csr.csr_matrix``` sparse array with all entries non-negative : ```A``` . Then ```A[i,j]``` represents the non-negative capacity of an edge from ```i'th```  vertex to the ```j'th``` vertex. 
```python
import maxflow
import numpy as np
from scipy import sparse

A = (sparse.rand(1000,1000,density=0.5,format='csr')*100).astype(np.uint32)
mflow1 = maxflow.dinic(A, 0, 999) # mflow1 contains the maximum flow computed by Dinic's algorithm with source being the 0'th node and sink being the last node.
maxflow.parallel_push_relabel(A, 0, 100, nthreads=3) # Runs Parallel push-relabel Algorithm using 3 OpenMP threads.
maxflow.ahuja_orlin(A.toarray(), 0, 120) # Runs on a Numpy array instead of a scipy sparse array.
```

## Benchmarks
All the algorithms are compared against Scipy's maximum flow implementation : ```scipy.sparse.csgraph._flow.maximum_flow``` with ```A``` (1000 x 1000) sampled using 
```python
A = (sparse.rand(1000,1000,density,format='csr')*100).astype(np.uint32)
```
All the algorithms except maxflow's edmond-karp's algorithm significantly outperforms scipy's implementation. The following table shows the average runtime of each algorithm:
|Algorithm | Density: 0.1 | Density: 0.3 | Density: 0.5 | Density: 0.9 |
|----------|--------------|--------------|--------------|--------------|
|Scipy's maximum_flow| 0.0605s | 0.2305s | 0.5193s | 1.1952s |
|ahuja_orlin | 0.0178s | 0.0545s | 0.0897s| 0.1159s |
|dinic | 0.0188s | 0.0561s | 0.0946s | 0.1202s |
|edmonds_karp | 0.0909s | 0.8998s | 2.5775s | 3.9351s |
|push_relabel_fifo | 0.0186s | 0.0559s | 0.0930s | 0.1214s |
|push_relabel_highest | 0.0179s | 0.0549s | 0.0931s | 0.1215s |
|parallel_AhujaOrlin_segment | 0.0181s | 0.0556s | 0.0896s | 0.1220s |
|parallel_push_relabel | 0.0188s | 0.0563s | 0.0942s | 0.1227s |
|parallel_push_relabel_segment | 0.0191s | 0.0572s | 0.0981s | 0.1319s |
