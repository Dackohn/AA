import time
import matplotlib.pyplot as plt
import numpy as np
import tracemalloc
import pandas as pd
from scipy.interpolate import make_interp_spline


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
    if method == 'iterative':
        return fibonacci_iterative(n)
    elif method == 'fast_doubling':
        return fibonacci_fast_doubling(n)
    else:
        raise ValueError("Unknown method")


def measure_time_and_memory(method):
    inputs = list(range(2000, 16001, 2000))
    times = []
    memory_usage = []
    results = []

    for n in inputs:
        tracemalloc.start()
        start_time = time.time()
        result = compute_fibonacci(n, method=method)
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(end_time - start_time)
        memory_usage.append(peak / 1024)  # Convert to KB
        results.append(result)

    return inputs, times, memory_usage, results[-1]


def print_results_table():
    methods = ['iterative', 'fast_doubling']

    results_dict = {}

    for method in methods:
        inputs, times, memory_usage, _ = measure_time_and_memory(method)
        results_dict[method] = times

    df = pd.DataFrame(results_dict, index=inputs).T
    df.columns = inputs

    print(df.to_string(index=True, float_format='%.6f'))


def plot_performance():
    methods = ['iterative', 'fast_doubling']

    for method in methods:
        inputs, times, memory_usage, last_result = measure_time_and_memory(method)
        inputs_smooth = np.linspace(min(inputs), max(inputs), 300)
        times_smooth = make_interp_spline(inputs, times)(inputs_smooth)
        memory_smooth = make_interp_spline(inputs, memory_usage)(inputs_smooth)

        plt.figure(figsize=(10, 6))
        plt.plot(inputs_smooth, times_smooth, label=f'Execution Time ({method.capitalize()})')
        plt.xlabel('n-th Fibonacci Term')
        plt.ylabel('Execution Time (seconds)')
        plt.title(f'Execution Time of {method.capitalize()} Method')
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(inputs_smooth, memory_smooth, label=f'Memory Usage ({method.capitalize()})', color='red')
        plt.xlabel('n-th Fibonacci Term')
        plt.ylabel('Memory Usage (KB)')
        plt.title(f'Memory Usage of {method.capitalize()} Method')
        plt.legend()
        plt.grid()
        plt.show()


plot_performance()
print_results_table()
