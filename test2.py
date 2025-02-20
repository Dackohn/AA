import time
import matplotlib.pyplot as plt
import numpy as np
import tracemalloc
from scipy.interpolate import make_interp_spline

def fibonacci_dynamic(n):
    if n <= 1:
        return n
    dp = [0, 1] + [0] * (n - 1)
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

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

def fibonacci_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def fibonacci_fast_doubling(n):
    def fib_dbl(m):
        if m == 0:
            return (0, 1)
        a, b = fib_dbl(m // 2)
        c = a * ((b << 1) - a)
        d = a * a + b * b
        if m % 2 == 0:
            return (c, d)
        else:
            return (d, c + d)
    return fib_dbl(n)[0]

def compute_fibonacci(n, method):
    if method == 'dynamic':
        return fibonacci_dynamic(n)
    #elif method == 'matrix':
    #    return fibonacci_matrix(n)
    elif method == 'iterative':
        return fibonacci_iterative(n)
    elif method == 'fast_doubling':
        return fibonacci_fast_doubling(n)
    else:
        raise ValueError("Unknown method")

def measure_time_and_memory(method):
    inputs = list(range(0, 16001, 2000))
    times = []
    memory_usage = []

    for n in inputs:
        tracemalloc.start()
        start_time = time.time()
        compute_fibonacci(n, method)
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(end_time - start_time)
        memory_usage.append(peak / 1024)  # Convert to KB

    return inputs, times, memory_usage

def plot_performance():
    methods = ['dynamic', 'iterative', 'fast_doubling']
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    for method in methods:
        inputs, times, memory_usage = measure_time_and_memory(method)
        inputs_smooth = np.linspace(min(inputs), max(inputs), 300)
        times_smooth = make_interp_spline(inputs, times)(inputs_smooth)
        memory_smooth = make_interp_spline(inputs, memory_usage)(inputs_smooth)

        axes[0].plot(inputs_smooth, times_smooth, label=f'Execution Time ({method.capitalize()})')
        axes[1].plot(inputs_smooth, memory_smooth, label=f'Memory Usage ({method.capitalize()})')

    axes[0].set_xlabel('n-th Fibonacci Term')
    axes[0].set_ylabel('Execution Time (seconds)')
    axes[0].set_title('Performance of Fibonacci Algorithms')
    axes[0].legend()
    axes[0].grid()

    axes[1].set_xlabel('n-th Fibonacci Term')
    axes[1].set_ylabel('Memory Usage (KB)')
    axes[1].set_title('Memory Usage of Fibonacci Algorithms')
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.show()

# Run performance plot
plot_performance()
