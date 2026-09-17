"""CPU workers with one numerical-library thread per process."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os

_limits = None


def configure_cpu_runtime():
    global _limits
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    from threadpoolctl import threadpool_limits

    _limits = threadpool_limits(limits=1)


def process_pool(workers):
    configure_cpu_runtime()
    return ProcessPoolExecutor(max_workers=workers,
                               mp_context=multiprocessing.get_context("spawn"),
                               initializer=configure_cpu_runtime)
