from multiprocessing.pool import ThreadPool
import numpy as np
import bottleneck as bn

pool = None
nthread = None


# ---------------------------------------------------------------------------
# functions that take one array as input

def o_nansum(arr, axis=None):
    return o_reduce(bn.nansum, arr, axis=axis)


def o_move_median(arr, window, min_count=None, axis=-1):
    return o_move(bn.move_median, arr, window=window, min_count=min_count,
                  axis=axis)


# ---------------------------------------------------------------------------
# functions that a list of arrays as input

def m_nansum(array_list, axis=None):
    return m_reduce(bn.nansum, array_list, axis=axis)


def m_move_median(array_list, window, min_count=None, axis=-1):
    return m_move(bn.move_median, array_list, window=window, min_count=min_count,
                  axis=axis)


# ---------------------------------------------------------------------------
# manage pool of threads

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

def o_reduce(func, arr, **kwargs):

    # check that pool is running
    if pool is None:
        raise ValueError("The thread pool has not been started")

    # maybe arr is not a ndarray; if so why use multithreading?!
    if type(arr) is np.ndarray:
        a = arr
    else:
        a = np.array(arr, copy=False)

    # axis
    axis = kwargs['axis']
    if axis is None:
        a = a.ravel()
        axis = 0
    elif axis < 0:
        axis += a.ndim
        if axis < 0:
            raise ValueError("axis(=%d) out of bounds" % axis)
    elif axis >= a.ndim:
        raise ValueError("axis(=%d) out of bounds" % axis)

    # the function can have only one input (arr); make it so
    unary_func = make_unary(func, **kwargs)

    # thread it!
    if a.ndim == 1:
        array_list = np.array_split(a, nthread, axis)
        out = unary_func(pool.map(unary_func, array_list))
    else:
        if axis == 0:
            split_axis = 1
        else:
            split_axis = 0
        array_list = np.array_split(a, nthread, split_axis)
        out_list = pool.map(unary_func, array_list)
        out = np.concatenate(out_list)

    return out


def o_move(func, arr, **kwargs):

    # check that pool is running
    if pool is None:
        raise ValueError("The thread pool has not been started")

    # maybe arr is not a ndarray; if so why use multithreading?!
    if type(arr) is np.ndarray:
        a = arr
    else:
        a = np.array(arr, copy=False)

    # axis
    axis = kwargs['axis']
    if axis is None:
        raise ValueError("An `axis` value of None is not supported.")
    elif axis < 0:
        axis += a.ndim
        if axis < 0:
            raise ValueError("axis(=%d) out of bounds" % axis)
    elif axis >= a.ndim:
        raise ValueError("axis(=%d) out of bounds" % axis)

    # the function can have only one input (arr); make it so
    unary_func = make_unary(func, **kwargs)

    # thread it!
    if a.ndim == 1:
        # moving window functions cannot be multi-threaded if ndim=1 (due
        # to edge effects), so run with a single thread
        out = unary_func(a)
    else:
        if axis == 0:
            split_axis = 1
        else:
            split_axis = 0
        array_list = np.array_split(a, nthread, split_axis)
        out_list = pool.map(unary_func, array_list)
        out = np.concatenate(out_list, split_axis)

    return out


def m_reduce(func, array_list, **kwargs):

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


def m_move(func, array_list, **kwargs):

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
