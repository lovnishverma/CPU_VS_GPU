import torch
import time
import torch.backends.cudnn as cudnn

# Improve cuDNN performance on Jetson Nano
cudnn.benchmark = True


def benchmark_matrix_multiplication(size=1000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Running matrix multiplication on {device.upper()} with size {size}x{size}...")

    A = torch.rand(size, size, device=device)
    B = torch.rand(size, size, device=device)

    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    C = torch.matmul(A, B)
    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Time taken on {device.upper()}: {elapsed_time:.4f} seconds")
    return elapsed_time


if __name__ == "__main__":
    benchmark_matrix_multiplication(size=1000)
