import maxflow
from scipy import sparse
import numpy as np
import time

THREADS = 3

def saveDimacs(x_array, filename):
    n = x_array.shape[0]
    m = np.count_nonzero(x_array)

    with open(filename, 'w') as f:
        f.write("p max {} {}\n".format(n, m))
        f.write("n {} {}\n".format(1, 's'))
        f.write("n {} {}\n".format(n, 't'))

        for i in range(n):
            for j in range(n):
                weight = x_array[i,j]
                if weight != 0:
                    f.write("a {} {} {}\n".format(i+1,j+1,weight))


def genMaxFlow(x_array):
    from ortools.graph import pywrapgraph
    n = x_array.shape[0]
    max_flow = pywrapgraph.SimpleMaxFlow()
    for i in range(0, n):
        for j in range(0, n):
            weight = int(x_array[i,j])
            if weight != 0:
                max_flow.AddArcWithCapacity(i, j, weight)

    return max_flow

def computeFlow(max_flow, n):
    max_flow.Solve(0, n-1)
    return max_flow.OptimalFlow()

def gen_matrix1(n, density):
    return (sparse.rand(n,n,density=density,format='csr')*100).astype(np.uint32)


def test_correctness(n, iters, seed=0, density=0.5, dense=True):
    print("------------Running test_correctness!--------------")
    print("n={}, iters={}, density={}, dense={}".format(n, iters, density, dense))
    fns_name = [m for m in dir(maxflow) if not m.startswith('__')]
    try:
        fns_name.remove('np')
        fns_name.remove('sparse')
    except Exception as e:
        pass
    fns = [eval('maxflow.'+i) for i in fns_name]
    print("functions testing:", fns_name)
    np.random.seed(seed)

    for i in range(iters):
        # x = np.random.randint(0, 2**32, size=(n,n), dtype=np.int32)
        x = (sparse.rand(n,n,density=density,format='csr')*200).astype(np.uint32)
        if dense:
            x_ = x.toarray()
        else:
            x_ = x
        flow = sparse.csgraph.maximum_flow(x, 0, n-1).flow_value
        for fn in fns:
            # print("u32:", fn.__name__)
            if 'parallel' in fn.__name__:
                flow2 = fn(x_, 0, n-1, THREADS)
            else:
                flow2 = fn(x_, 0, n-1)
            assert flow == flow2, "Error at iteration:{}! : function {} gives flow:{}, while scipy's maxflow:{}".format(i, fn.__name__, flow2, flow)

        x = (sparse.rand(n,n,density=density,format='csr')*2**30).astype(np.uint64)
        flow = sparse.csgraph.maximum_flow(x, 0, n-1).flow_value
        # flow_or = computeFlow(genMaxFlow(x_), n)
        flow_or = 0
        if dense:
            x_ = x.toarray()
        else:
            x_ = x
        for fn in fns:
            if 'parallel' in fn.__name__:
                flow2 = fn(x_, 0, n-1, THREADS)
            else:
                flow2 = fn(x_, 0, n-1)
            assert flow == flow2, "Error at iteration:{}! : function {} gives flow:{}, while scipy's maxflow:{}, OR:{}".format(i, fn.__name__, flow2, flow, flow_or)


def bench(n, iters, seed=0, density=0.5, dense=True, matrix_generator=gen_matrix1):
    print("------------Running bench!--------------")
    print("n={}, iters={}, density={}, dense={}".format(n, iters, density, dense))
    fns_name = [m for m in dir(maxflow) if not m.startswith('__')]
    try:
        fns_name.remove('np')
        fns_name.remove('sparse')
    except Exception as e:
        pass
    fns = [eval('maxflow.'+i) for i in fns_name]
    print("functions testing:", fns_name)
    np.random.seed(seed)

    scipy_time = 0
    maxflow_times = np.zeros(len(fns), dtype=np.float64)

    for i in range(iters):
        # x = np.random.randint(0, 2**32, size=(n,n), dtype=np.int32)
        x = matrix_generator(n, density)
        if dense:
            x_ = x.toarray()
        else:
            x_ = x
        t1 = time.perf_counter()
        flow = sparse.csgraph.maximum_flow(x, 0, n-1).flow_value
        scipy_time += (time.perf_counter() - t1)
        for i, fn in enumerate(fns):
            if 'parallel' in fn.__name__:
                t1 = time.perf_counter()
                flow2 = fn(x_, 0, n-1, THREADS)
                maxflow_times[i] += (time.perf_counter() - t1)
            else:
                t1 = time.perf_counter()
                flow2 = fn(x_, 0, n-1)
                maxflow_times[i] += (time.perf_counter() - t1)
            assert flow == flow2, "Error at iteration:{}! : function {} gives flow:{}, while scipy's maxflow:{}".format(i, fn.__name__, flow2, flow)

    print("Average time taken for scipy:", scipy_time/iters)
    avg_maxflow_times = maxflow_times/iters
    print("Average time taken for maxflow algs:", list(zip(fns_name, list(avg_maxflow_times))))




test_correctness(100, 100, dense=True)
test_correctness(100, 100, dense=False)

bench(1000,10, density=0.1, dense=True, matrix_generator=gen_matrix1)
bench(1000,10, density=0.1, dense=False, matrix_generator=gen_matrix1)

bench(1000,10, density=0.3, dense=True, matrix_generator=gen_matrix1)
bench(1000,10, density=0.3, dense=False, matrix_generator=gen_matrix1)

bench(1000,10, density=0.5, dense=True, matrix_generator=gen_matrix1)
bench(1000,10, density=0.5, dense=False, matrix_generator=gen_matrix1)

bench(1000,10, density=0.9, dense=True, matrix_generator=gen_matrix1)
bench(1000,10, density=0.9, dense=False, matrix_generator=gen_matrix1)
