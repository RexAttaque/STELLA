"""CPU threads of the compiled (numba) parallel kernels.

The number of threads is a setting of the detection tab (0 = automatic). Automatic leaves three logical CPUs to
the threads that work beside the kernels: the two threads preparing the next detection steps (time window and
labelling) and the GUI. That is faster than taking every CPU: numba's OpenMP threads keep spinning ~0.1 s after
each parallel loop (then sleep), and a detection step runs several short loops, so with every CPU taken the
preparing threads would wait for a core. On an 8-core / 16-thread CPU, 553 M events: 16 threads with passive
waiting 32.3 s, 13 threads 28.7 s.

numba applies the number per calling thread, so code running in a worker thread (the detection script) calls
apply() first. Every module with parallel kernels imports this one.
"""
import numba

RESERVED = 3  # logical CPUs left to the step-preparing threads and the GUI when automatic

_threads = 0  # 0: automatic


def available():
    """Threads numba can use (logical CPUs, or NUMBA_NUM_THREADS)."""
    return int(numba.config.NUMBA_NUM_THREADS)


def auto():
    return max(1, available() - RESERVED)


def set_threads(n):
    """Threads for the parallel kernels from now on (0 or None: automatic); returns the number in use."""
    global _threads
    _threads = max(0, min(int(n or 0), available()))
    apply()
    return threads()


def threads():
    return _threads or auto()


def apply():
    """Applies the setting to the calling thread (numba's thread count is per thread)."""
    numba.set_num_threads(threads())
