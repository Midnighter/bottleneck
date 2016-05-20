from multiprocessing.pool import ThreadPool
import bottleneck as bn

pool = None
nthread = None


def m_nansum(array_list, axis=None):
    return reduces(bn.nansum, array_list, axis=axis)


def m_move_median(array_list, window, min_count=None, axis=-1):
    return move(bn.move_median, array_list, window=window, min_count=min_count,
                axis=axis)


# ---------------------------------------------------------------------------
# start and stop pool of threads

def start_pool(nthreads=None):
    global pool
    global nthread
    stop_pool()
    pool = ThreadPool(nthreads)
    nthread = nthreads


def stop_pool():
    global pool
    global nthread
    if pool is not None:
        pool.pool.terminate()
        pool = None
    nthread = None


# ---------------------------------------------------------------------------
# utility functions

def reduces(func, array_list, **kwargs):

    # check that pool is running
    if pool is None:
        raise ValueError("The thread pool has not been started")

    # axis
    axis = kwargs['axis']
    if axis is None:
        array_list = [a.ravel() for a in array_list]
        axis = 0

    # the function can have only one input (arr); make it so
    unary_func = make_unary(func, **kwargs)

    # thread it!
    out_list = pool.map(unary_func, array_list)

    return out_list


def move(func, array_list, **kwargs):

    # check that pool is running
    if pool is None:
        raise ValueError("The thread pool has not been started")

    # axis
    axis = kwargs['axis']
    if axis is None:
        raise ValueError("An `axis` value of None is not supported.")

    # the function can have only one input (arr); make it so
    unary_func = make_unary(func, **kwargs)

    # thread it!
    out_list = pool.map(unary_func, array_list)

    return out_list


def make_unary(func, **kwargs):
    def unary_func(arr):
        return func(arr, **kwargs)
    return unary_func
