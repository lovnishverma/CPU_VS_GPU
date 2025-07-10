# matrix.py

# PyTorch CPU vs GPU Benchmark

A comprehensive benchmarking tool to compare matrix multiplication performance between CPU and GPU using PyTorch. This tool provides detailed performance analysis including timing statistics, FLOPS calculations, and multi-size benchmarking capabilities.

## Features

- 🔵 **CPU Benchmarking**: Measures matrix multiplication performance on CPU
- 🟢 **GPU Benchmarking**: Measures matrix multiplication performance on CUDA GPU
- 📊 **Multi-Size Testing**: Benchmark across different matrix sizes
- 📈 **Performance Metrics**: Calculate and display FLOPS (Floating Point Operations Per Second)
- 🎯 **Statistical Analysis**: Multiple runs with mean and standard deviation
- 🔧 **Flexible Configuration**: Command-line arguments for customization
- 🖥️ **System Information**: Display PyTorch version, CUDA version, and GPU specifications

## Requirements

- Python 3.6+
- PyTorch
- CUDA-compatible GPU (optional, for GPU benchmarking)

## Installation

1. Install PyTorch:
```bash
# CPU only
pip install torch

# With CUDA support (replace cu118 with your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

2. Download the benchmark script:
```bash
# Using wget
wget https://raw.githubusercontent.com/lovnishverma/CPU_VS_GPU/refs/heads/main/matrix.py

# Or using curl
curl -O https://raw.githubusercontent.com/lovnishverma/CPU_VS_GPU/refs/heads/main/matrix.py

# Or simply copy the script from the repository
```

## Usage

### Basic Usage

Run a single benchmark with default settings (10000x10000 matrix, 5 runs):
```bash
python benchmark.py
```

### Custom Matrix Size

Benchmark with a specific matrix size:
```bash
python benchmark.py --size 5000
```

### Multiple Runs

Increase the number of runs for better statistical accuracy:
```bash
python benchmark.py --runs 10
```

### Multi-Size Benchmarking

Test performance across multiple matrix sizes:
```bash
python benchmark.py --multi-size
```

Custom sizes for multi-size benchmarking:
```bash
python benchmark.py --multi-size --sizes 1000 2000 5000 10000 15000
```

### Quick Single Run

For quick testing:
```bash
python benchmark.py --runs 1
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--size` | int | 10000 | Matrix size for single benchmark |
| `--runs` | int | 5 | Number of runs for statistical analysis |
| `--multi-size` | flag | False | Enable multi-size benchmarking |
| `--sizes` | list | [1000, 2000, 5000, 10000] | Matrix sizes for multi-size benchmark |

## Sample Output

### Single Size Benchmark
```
PS C:\Users\princ\Documents\GPU Vs CPU\GPU> python matrix.py     
===== 🚀 PyTorch Performance Benchmark =====
PyTorch version: 2.7.0+cu118
CUDA available: True
CUDA version: 11.8
GPU device: NVIDIA GeForce RTX 2050
GPU memory: 4.0 GB

[🔵 Torch CPU] Matrix size: 10000x10000
[CPU] Run 1: 11.8188 seconds
[CPU] Run 2: 11.1280 seconds
[CPU] Run 3: 11.9454 seconds
[CPU] Run 4: 10.6777 seconds
[CPU] Run 5: 10.2945 seconds
[CPU] Average time: 11.1729 ± 0.7129 seconds

[🟢 Torch GPU] Matrix size: 10000x10000
[GPU] Device: NVIDIA GeForce RTX 2050
[GPU] Warming up...
[GPU] Starting timed multiplication...
[GPU] Run 1: 0.5288 seconds
[GPU] Run 2: 0.5319 seconds
[GPU] Run 3: 0.5157 seconds
[GPU] Run 4: 0.5213 seconds
[GPU] Run 5: 0.5159 seconds
[GPU] Average time: 0.5227 ± 0.0074 seconds

===== ⏱️ Performance Summary =====
CPU Time: 11.1729 ± 0.7129 s
CPU Performance: 179.00 GFLOPS
GPU Time: 0.5227 ± 0.0074 s
GPU Performance: 3.83 TFLOPS
Speedup: 21.38x faster on GPU
```

### Multi-Size Benchmark
```
===== 📈 Performance Summary =====
Size     CPU Time     GPU Time     Speedup    CPU FLOPS    GPU FLOPS   
--------------------------------------------------------------------------------
1000     0.0080       0.0011       7.41x      248.61 GFLOPS  1.84 TFLOPS
2000     0.0932       0.0063       14.78x     171.60 GFLOPS  2.54 TFLOPS
5000     1.5635       0.0823       19.00x     159.88 GFLOPS  3.04 TFLOPS
10000    11.3886      0.5285       21.55x     175.61 GFLOPS  3.78 TFLOPS
15000    42.6241      21.4259      1.99x      158.36 GFLOPS  315.03 GFLOPS
```

## Understanding the Results

### Performance Metrics

- **Time**: Execution time in seconds (lower is better)
- **FLOPS**: Floating Point Operations Per Second (higher is better)
- **Speedup**: How many times faster GPU is compared to CPU

### Expected Performance Patterns

- **Small matrices (< 2000x2000)**: CPU may be competitive, GPU shows 7-15x speedup
- **Medium matrices (2000-10000)**: GPU shows consistent 15-22x speedup advantage
- **Large matrices (> 10000)**: GPU memory limitations may reduce performance gains
- **Very large matrices (> 15000)**: Memory bandwidth becomes bottleneck, speedup drops significantly
- **Typical speedups**: 2-25x depending on matrix size and GPU memory capacity

### FLOPS Calculation

The script calculates FLOPS using the formula for matrix multiplication:
- Operations = 2 × n³ - n² (where n is matrix size)
- FLOPS = Operations ÷ Time

## Troubleshooting

### CUDA Not Available

If you see "CUDA GPU not available":
1. Ensure you have a CUDA-compatible GPU
2. Install PyTorch with CUDA support
3. Check CUDA installation: `nvidia-smi`

### Memory Issues

For large matrices, you might encounter out-of-memory errors or performance degradation:
- **GPU Memory Limitation**: RTX 2050 has 4GB VRAM - matrices > 12000x12000 may cause issues
- Reduce matrix size using `--size`
- Use smaller sizes in multi-size benchmark
- Close other GPU applications
- **Performance Drop**: Very large matrices (>15000x15000) show reduced speedup due to memory bandwidth limits

### Performance Inconsistency

For more consistent results:
- Increase number of runs: `--runs 10`
- Close other applications
- Ensure GPU is not thermal throttling

## Technical Details

### Benchmarking Methodology

1. **Matrix Generation**: Random matrices generated on respective devices
2. **GPU Warm-up**: 3 warm-up iterations to eliminate initialization overhead
3. **Synchronization**: Proper CUDA synchronization for accurate GPU timing
4. **Statistical Analysis**: Multiple runs with mean and standard deviation

### Hardware Considerations

- **GPU Architecture**: Newer architectures (Ampere, Ada Lovelace) show better performance
- **Memory Bandwidth**: Higher bandwidth GPUs perform better
- **CPU**: Modern CPUs with AVX instructions perform better
- **System Memory**: Sufficient RAM prevents swapping


---


## image.py

# ⚡ PyTorch Inference Benchmark: CPU vs GPU

This Python script benchmarks the inference performance of pretrained CNN models from `torchvision` on both **CPU** and **GPU**, allowing you to compare speed, throughput, and predictions side-by-side.

---

## 📌 Features

- 🔁 Compare CPU vs GPU inference times
- 🧠 Benchmark ResNet-18, ResNet-50, MobileNet V3
- 📈 Prints detailed performance metrics:
  - Average, min, max time
  - Throughput (images/sec)
- ✅ Batch size and run count configurable
- 💾 Optional logging to JSON

---

## 📦 Requirements

```bash
pip install torch torchvision pillow
````

---

## 🚀 Usage

```bash
python image.py --image sample.jpg
```

### Optional arguments:

| Argument       | Description                                                | Default       |
| -------------- | ---------------------------------------------------------- | ------------- |
| `--image`      | Path to input image (must exist)                           | **required**  |
| `--model`      | Model to benchmark: `resnet18`, `resnet50`, `mobilenet_v3` | `resnet18`    |
| `--runs`       | Number of inference runs                                   | `10`          |
| `--batch-size` | Inference batch size                                       | `1`           |
| `--save`       | Save results to `logs/benchmark_results.json`              | `False` (off) |

---

## ✅ Example Output

```text
===== 🔁 CPU vs GPU Inference Comparison =====

🔍 Device: CPU
📦 Model: ResNet-18 (11M parameters)
🔢 Batch size: 1, 🔄 Runs: 10

📊 CPU Results:
  Avg: 0.0367s ± 0.0009s, Min: 0.0357s, Max: 0.0382s
  Throughput: 27.26 images/sec
  Predicted: revolver

🔍 Device: NVIDIA GeForce RTX 2050
📦 Model: ResNet-18 (11M parameters)
🔢 Batch size: 1, 🔄 Runs: 10

📊 NVIDIA GeForce RTX 2050 Results:
  Avg: 0.0036s ± 0.0004s, Min: 0.0034s, Max: 0.0044s
  Throughput: 275.70 images/sec
  Predicted: revolver

===== 📈 Comparison Summary =====
Device               Avg Time     Throughput      Predicted Class
----------------------------------------------------------------------
CPU                  0.0367       27.26           revolver
NVIDIA GeForce RTX 2050 0.0036       275.70          revolver
```

---

## 📂 Output

If you use `--save`, results will be stored in:

```
logs/benchmark_results.json
```

Including metadata like:

* Timestamp
* PyTorch & CUDA version
* Device name
* Run statistics

---

## 🖼 Image Requirements

Input image must:

* Exist at the given path
* Be RGB-compatible (e.g., `.jpg`, `.png`)

If not found, the script will raise an error.

---

## 🔧 Future Improvements

* ✅ Graph plotting (matplotlib)
* ✅ CSV export
* 🧪 More models (EfficientNet, ViT)
* 🧠 Quantized models / ONNX runtime

---

## 📜 License

MIT License

---

## 🙌 Acknowledgements

* [PyTorch](https://pytorch.org/)
* [Torchvision Models](https://pytorch.org/vision/stable/models.html)

```

