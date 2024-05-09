from timeit import timeit
# py=timeit('fib(93)',number=1_000_000,setup='from teste import fib')
cython=timeit('fib(93)',number=5_000_000,setup='from cythonfib import fib ')
px = timeit('fib(93)', number=5_000_000, setup='from fib_x import fib')
# print(f"time Python:{py}")
print(f"time Puro:{px}")
print(f"time cython:{cython}")
# print(py/cython)
# print(py/px)