"""Shows the effect of releasing the GIL during triangulation"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import PythonCDT as cdt

N_THREADS = min(8, os.cpu_count())
N_VERTICES = 200_000


def triangulate(vertices) -> None:
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AUTO, cdt.IntersectingConstraintEdges.NOT_ALLOWED, 0.0)
    t.insert_vertices(vertices)


def seconds(f) -> float:
    start = time.perf_counter()
    f()
    return time.perf_counter() - start


rng = np.random.default_rng(0)
batches = [rng.random((N_VERTICES, 2)) for _ in range(N_THREADS)]

# separate triangulations built in parallel threads
sequential = seconds(lambda: [triangulate(b) for b in batches])
with ThreadPoolExecutor(N_THREADS) as pool:
    threaded = seconds(lambda: list(pool.map(triangulate, batches)))
print(f"{N_THREADS} triangulations of {N_VERTICES} vertices each")
print(f"  sequential {sequential:8.3f} s")
print(f"  {N_THREADS} threads  {threaded:8.3f} s  ({sequential / threaded:.1f}x speedup)")

# a Python thread keeps running while another thread triangulates
worker = threading.Thread(target=triangulate, args=(batches[0],))
ticks = [time.perf_counter()]
worker.start()
while worker.is_alive():
    time.sleep(0.001)
    ticks.append(time.perf_counter())
longest_stall = max(np.diff(ticks))
print(f"main thread ticking every 1 ms during {ticks[-1] - ticks[0]:.3f} s triangulation")
print(f"  ticks {len(ticks) - 1:>6}, longest stall {1e3 * longest_stall:8.1f} ms")
