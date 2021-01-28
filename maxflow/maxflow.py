from ._maxflow import (
                        _ahuja_orlin_dense, 
                        _ahuja_orlin_sparse, 
                        _dinic_dense, 
                        _dinic_sparse, 
                        _edmonds_karp_dense,
                        _edmonds_karp_sparse,
                        _parallel_AhujaOrlin_segment_dense,
                        _parallel_AhujaOrlin_segment_sparse,
                        _parallel_push_relabel_dense,
                        _parallel_push_relabel_sparse,
                        _parallel_push_relabel_segment_dense,
                        _parallel_push_relabel_segment_sparse,
                        _push_relabel_fifo_dense,
                        _push_relabel_fifo_sparse,
                        _push_relabel_highest_dense,
                        _push_relabel_highest_sparse,
                        _destroy_graph
                        )


import types
import numpy as np
from scipy import sparse

_alg_params = {
                "edmonds_karp"                  : (1, _edmonds_karp_dense, _edmonds_karp_sparse, False), 
                "ahuja_orlin"                   : (2, _ahuja_orlin_dense, _ahuja_orlin_sparse, False),
                "dinic"                         : (1, _dinic_dense, _dinic_sparse, False),
                "push_relabel_fifo"             : (2, _push_relabel_fifo_dense, _push_relabel_fifo_sparse, False),
                "push_relabel_highest"          : (2, _push_relabel_highest_dense, _push_relabel_highest_sparse, False),
                "parallel_push_relabel"         : (2, _parallel_push_relabel_dense, _parallel_push_relabel_sparse, True),
                "parallel_push_relabel_segment" : (2, _parallel_push_relabel_segment_dense, _parallel_push_relabel_segment_sparse, True),
                "parallel_AhujaOrlin_segment"   : (2, _parallel_AhujaOrlin_segment_dense, _parallel_AhujaOrlin_segment_sparse, True),
                }


def get_alg_names():
    return list(_alg_params.keys())


class Solver:
    def __init__(self):
        self.__graph_idx = None
        self.__mode = None
        self.alg = None
        self.isLoaded = False

    def load_graph(self, A, alg):
        if alg not in _alg_params:
            raise ValueError("alg must be any one of : [{}]".format(', '.join(_alg_params)))
        
        self.__mode = get_mode(A)
        self.__func_idx = 1
        
        if type(A) is sparse.csr.csr_matrix:
            self.__func_idx = 2

        if self.isLoaded:
            self.destroy_graphs()

        if self.__func_idx == 1:
            self.n = A.shape[0]
            self.__graph_idx = _alg_params[alg][1](self.__mode, A.ctypes.data, self.n, 0, 0, -2)[1]
        else:
            A_coo = A.tocoo(copy=False)
            self.n, m = A_coo.shape[0], A_coo.nnz
            self.__graph_idx = _alg_params[alg][2](self.__mode, A_coo.data.ctypes.data, A_coo.row.ctypes.data, A_coo.col.ctypes.data, self.n, m, 0, 0, -2)[1]
    
        self.isLoaded = True
        self.alg = alg

    def solve(self, alg, source, sink, nthreads=1):
        if (alg != self.alg) or (not self.isLoaded):
            raise ValueError("Run load_graph method first using alg as a parameter first")

        if (type(source) != int) or (not (0 <= source < self.n)):
            raise ValueError("source must be a non-negative integer smaller than number of vertices")

        if (type(sink) != int) or (not (0 <= sink < self.n)):
            raise ValueError("sink must be a non-negative integer smaller than number of vertices")

        if source == sink:
            raise ValueError("source and sink must be different vertices")

        args = [self.__mode, 0, 0, source, sink, self.__graph_idx] if (self.__func_idx == 1) else [self.__mode, 0, 0, 0, 0, 0, source, sink, self.__graph_idx]
        param = _alg_params[alg]
        if param[3]:
            args.append(nthreads)
        
        flow_value = param[self.__func_idx](*args)[0]
        self.destroy_graphs()
        return flow_value

    def destroy_graphs(self):
        if self.__graph_idx is not None:
            num_delete = _destroy_graph(self.__graph_idx)
            assert num_delete == 1, "1 graph should be deleted!" 
            self.__graph_idx = None
            self.__mode = None
        self.isLoaded = False

    def __del__(self):
        self.destroy_graphs()


def get_mode(A):
    uint32_max = np.iinfo(np.uint32).max
    if type(A) not in (np.ndarray, sparse.csr.csr_matrix):
        raise TypeError("A must be either numpy.ndarray or scipy.sparse.csr.csr_matrix")

    if A.shape[0] != A.shape[1]:
        raise ValueError("A.shape[0] != A.shape[1] : A must be square")

    if A.shape[0] == 1:
        raise ValueError("The graph must have more than one vertex : A.shape[0] must be greater than 1")

    if A.shape[0] <= uint32_max:
        if A.dtype < np.int64:
            return 1
        else:
            return 2
    else:
        if A.dtype < np.int64:
            return 3
        else:
            return 4


def create_maxflow_runner(fn_dense, fn_sparse, isThreaded):

    def maxflow_computer_sequential(A, source, sink):
        mode = get_mode(A)

        if type(A) is sparse.csr.csr_matrix:
            A_coo = A.tocoo(copy=False)
            n,m = A_coo.shape[0], A_coo.nnz
            params = [mode, A_coo.data.ctypes.data, A_coo.row.ctypes.data, A_coo.col.ctypes.data, n, m, source, sink, -1]
            return fn_sparse(*params)[0]

        elif type(A) is np.ndarray:
            np.ascontiguousarray(A)
            n = A.shape[0]
            params = [mode, A.ctypes.data, n, source, sink, -1]
            return fn_dense(*params)[0]

    def maxflow_computer_parallel(A, source, sink, nthreads=1):
        mode = get_mode(A)

        if type(A) is sparse.csr.csr_matrix:
            A_coo = A.tocoo(copy=False)
            n,m = A_coo.shape[0], A_coo.nnz
            params = [mode, A_coo.data.ctypes.data, A_coo.row.ctypes.data, A_coo.col.ctypes.data, n, m, source, sink, -1, nthreads]
            return fn_sparse(*params)[0]

        elif type(A) is np.ndarray:
            np.ascontiguousarray(A)
            n = A.shape[0]
            params = [mode, A.ctypes.data, n, source, sink, -1, nthreads]
            return fn_dense(*params)[0]

    if isThreaded:
        return maxflow_computer_parallel
    else:
        return maxflow_computer_sequential 


for name, alg in _alg_params.items():
    cmd = "{} = create_maxflow_runner({}, {}, {})".format(name, alg[1].__name__, alg[2].__name__, alg[3]) 
    exec(cmd)

