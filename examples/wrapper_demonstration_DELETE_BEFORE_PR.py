from time import perf_counter

import numpy as np

from floris.utilities import split_calculate_join


@split_calculate_join
def big_math(x, y, z):
    # return (x ** y) / (y ** z)
    return np.cbrt(np.mean((x) ** 3)) * y ** z


def big_math_no_split(x, y, z):
    return np.cbrt(np.mean((x) ** 3)) * y ** z


x = np.random.random((720, 80, 100))
y = np.random.random((720, 80, 100))
z = np.random.random((720, 80, 100))


start = perf_counter()
res1 = big_math(x, y, z)
end = perf_counter()
print(f"    Split: {end - start:.6f}")

start = perf_counter()
res2 = big_math_no_split(x, y, z)
end = perf_counter()
print(f"Not split: {end - start:.6f}")

print(f"Results are equal: {np.all(res1 == res2)}")
