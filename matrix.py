import torch
import time
import statistics
import argparse


def benchmark_torch_cpu(size=10000, num_runs=5):
    print(f"\n[🔵 Torch CPU] Matrix size: {size}x{size}")
    A = torch.rand(size, size, device="cpu")
    B = torch.rand(size, size, device="cpu")

    times = []
    for i in range(num_runs):
        start = time.time()
        C = torch.matmul(A, B)
        end = time.time()
        times.append(end - start)
        if num_runs > 1:
            print(f"[CPU] Run {i+1}: {times[-1]:.4f} seconds")

    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0

    print(f"[CPU] Average time: {avg_time:.4f} ± {std_time:.4f} seconds")
    return avg_time, std_time


def benchmark_torch_gpu(size=10000, num_runs=5):
    if not torch.cuda.is_available():
        print("\n[🔴 Torch GPU] CUDA GPU not available.")
        return None, None

    print(f"\n[🟢 Torch GPU] Matrix size: {size}x{size}")
    print(f"[GPU] Device: {torch.cuda.get_device_name()}")

    A = torch.rand(size, size, device="cuda")
    B = torch.rand(size, size, device="cuda")

    # Warm-up GPU to remove init overhead
    print("[GPU] Warming up...")
    for _ in range(3):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()

    # Benchmark
    print("[GPU] Starting timed multiplication...")
    times = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
        if num_runs > 1:
            print(f"[GPU] Run {i+1}: {times[-1]:.4f} seconds")

    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0

    print(f"[GPU] Average time: {avg_time:.4f} ± {std_time:.4f} seconds")
    return avg_time, std_time


def calculate_flops(size, time_seconds):
    """Calculate FLOPS (Floating Point Operations Per Second)"""
    # Matrix multiplication: 2 * n^3 - n^2 operations
    ops = 2 * size**3 - size**2
    flops = ops / time_seconds
    return flops


def format_flops(flops):
    """Format FLOPS in human-readable units"""
    if flops >= 1e12:
        return f"{flops/1e12:.2f} TFLOPS"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f} GFLOPS"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f} MFLOPS"
    else:
        return f"{flops:.2f} FLOPS"


def benchmark_multiple_sizes(sizes, num_runs=3):
    """Benchmark multiple matrix sizes"""
    print("\n===== 📊 Multi-Size Benchmark =====")
    results = []

    for size in sizes:
        print(f"\n--- Matrix Size: {size}x{size} ---")
        cpu_time, cpu_std = benchmark_torch_cpu(size, num_runs)
        gpu_time, gpu_std = benchmark_torch_gpu(size, num_runs)

        cpu_flops = calculate_flops(size, cpu_time)
        gpu_flops = calculate_flops(size, gpu_time) if gpu_time else 0

        results.append({
            'size': size,
            'cpu_time': cpu_time,
            'cpu_std': cpu_std,
            'gpu_time': gpu_time,
            'gpu_std': gpu_std,
            'cpu_flops': cpu_flops,
            'gpu_flops': gpu_flops
        })

    # Print summary table
    print("\n===== 📈 Performance Summary =====")
    print(f"{'Size':<8} {'CPU Time':<12} {'GPU Time':<12} {'Speedup':<10} {'CPU FLOPS':<12} {'GPU FLOPS':<12}")
    print("-" * 80)

    for result in results:
        size = result['size']
        cpu_time = result['cpu_time']
        gpu_time = result['gpu_time']
        cpu_flops = format_flops(result['cpu_flops'])
        gpu_flops = format_flops(result['gpu_flops']) if gpu_time else "N/A"
        speedup = f"{cpu_time/gpu_time:.2f}x" if gpu_time else "N/A"

        print(f"{size:<8} {cpu_time:<12.4f} {gpu_time if gpu_time else 'N/A':<12} {speedup:<10} {cpu_flops:<12} {gpu_flops:<12}")


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch CPU vs GPU Benchmark')
    parser.add_argument('--size', type=int, default=10000,
                        help='Matrix size (default: 10000)')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of runs (default: 5)')
    parser.add_argument('--multi-size', action='store_true',
                        help='Benchmark multiple sizes')
    parser.add_argument('--sizes', nargs='+', type=int, default=[1000, 2000, 5000, 10000],
                        help='Sizes for multi-size benchmark')

    args = parser.parse_args()

    print("===== 🚀 PyTorch Performance Benchmark =====")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name()}")
        print(
            f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    if args.multi_size:
        benchmark_multiple_sizes(args.sizes, args.runs)
    else:
        cpu_time, cpu_std = benchmark_torch_cpu(args.size, args.runs)
        gpu_time, gpu_std = benchmark_torch_gpu(args.size, args.runs)

        print("\n===== ⏱️ Performance Summary =====")
        print(f"CPU Time: {cpu_time:.4f} ± {cpu_std:.4f} s")
        print(
            f"CPU Performance: {format_flops(calculate_flops(args.size, cpu_time))}")

        if gpu_time:
            print(f"GPU Time: {gpu_time:.4f} ± {gpu_std:.4f} s")
            print(
                f"GPU Performance: {format_flops(calculate_flops(args.size, gpu_time))}")
            print(f"Speedup: {cpu_time / gpu_time:.2f}x faster on GPU")
        else:
            print("GPU benchmark skipped due to unavailable device.")


if __name__ == "__main__":
    main()
