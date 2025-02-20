import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline


def matrix_mult(A, B):
    return [[A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
            [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]]


def matrix_power(F, n):
    result = [[1, 0], [0, 1]]  # Identity matrix
    base = F
    while n > 0:
        if n % 2 == 1:
            result = matrix_mult(result, base)
        base = matrix_mult(base, base)
        n //= 2
    return result


def fibonacci_matrix(n):
    if n == 0:
        return 0
    F = [[1, 1], [1, 0]]
    result = matrix_power(F, n - 1)
    return result[0][0]


def measure_time_matrix():
    inputs = list(range(0, 300000, 37500))
    times = []

    for n in inputs:
        start_time = time.time()
        fibonacci_matrix(n)
        end_time = time.time()
        times.append(end_time - start_time)

    return inputs, times


def print_results_table():
    inputs, times = measure_time_matrix()
    df = pd.DataFrame([times], columns=inputs, index=["Matrix Power Method"])
    print(df.to_string(index=True, float_format='%.6f'))


def plot_matrix_performance():
    inputs, times = measure_time_matrix()
    inputs_smooth = np.linspace(min(inputs), max(inputs), 300)
    times_smooth = make_interp_spline(inputs, times)(inputs_smooth)

    plt.figure(figsize=(10, 6))
    plt.plot(inputs_smooth, times_smooth, label='Matrix Power Method')
    plt.xlabel('n-th Fibonacci Term')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time of Fibonacci Matrix Power Method')
    plt.legend()
    plt.grid()
    plt.show()


# Run performance plot and print results
plot_matrix_performance()
print_results_table()
