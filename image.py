import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights, MobileNet_V3_Large_Weights
from PIL import Image
import os
import argparse
import statistics
import json
from datetime import datetime


def get_model_info():
    return {
        'resnet18': {
            'model_fn': models.resnet18,
            'weights': ResNet18_Weights.DEFAULT,
            'description': 'ResNet-18 (11M parameters)'
        },
        'resnet50': {
            'model_fn': models.resnet50,
            'weights': ResNet50_Weights.DEFAULT,
            'description': 'ResNet-50 (25M parameters)'
        },
        'mobilenet_v3': {
            'model_fn': models.mobilenet_v3_large,
            'weights': MobileNet_V3_Large_Weights.DEFAULT,
            'description': 'MobileNet V3 Large (5.5M parameters)'
        }
    }


def run_inference_benchmark(image_path, model_name="resnet18", device_str="cpu", num_runs=10, batch_size=1):
    model_info = get_model_info()
    if model_name not in model_info:
        raise ValueError(f"Model {model_name} not supported.")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    device = torch.device("cuda" if device_str ==
                          "gpu" and torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(
        0) if device.type == "cuda" else "CPU"

    print(f"\n🔍 Device: {device_name}")
    print(f"📦 Model: {model_info[model_name]['description']}")
    print(f"🔢 Batch size: {batch_size}, 🔄 Runs: {num_runs}")

    weights = model_info[model_name]['weights']
    model = model_info[model_name]['model_fn'](weights=weights).to(device)
    model.eval()

    transform = weights.transforms()
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)
    if batch_size > 1:
        input_tensor = input_tensor.repeat(batch_size, 1, 1, 1)
    input_tensor = input_tensor.to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()

    times = []
    for run in range(num_runs):
        with torch.no_grad():
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            output = model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()

            times.append(end_time - start_time)

    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    min_time = min(times)
    max_time = max(times)
    throughput = batch_size / avg_time

    predicted_class = weights.meta["categories"][output[0].argmax().item()]

    results = {
        'device': device_name,
        'model': model_name,
        'batch_size': batch_size,
        'num_runs': num_runs,
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'throughput': throughput,
        'predicted_class': predicted_class
    }

    print(f"\n📊 {device_name} Results:")
    print(
        f"  Avg: {avg_time:.4f}s ± {std_time:.4f}s, Min: {min_time:.4f}s, Max: {max_time:.4f}s")
    print(f"  Throughput: {throughput:.2f} images/sec")
    print(f"  Predicted: {predicted_class}")

    return results


def compare_cpu_gpu(image_path, model_name="resnet18", num_runs=10, batch_size=1):
    print("\n===== 🔁 CPU vs GPU Inference Comparison =====")
    results = []

    cpu_result = run_inference_benchmark(
        image_path, model_name, device_str="cpu", num_runs=num_runs, batch_size=batch_size)
    results.append(cpu_result)

    if torch.cuda.is_available():
        gpu_result = run_inference_benchmark(
            image_path, model_name, device_str="gpu", num_runs=num_runs, batch_size=batch_size)
        results.append(gpu_result)
    else:
        print("\n🚫 GPU not available. Skipping GPU benchmark.")

    print("\n===== 📈 Comparison Summary =====")
    print(f"{'Device':<20} {'Avg Time':<12} {'Throughput':<15} {'Predicted Class':<20}")
    print("-" * 70)
    for result in results:
        print(
            f"{result['device']:<20} {result['avg_time']:<12.4f} {result['throughput']:<15.2f} {result['predicted_class']:<20}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compare PyTorch Inference on CPU and GPU')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model', type=str, default='resnet18', choices=[
                        'resnet18', 'resnet50', 'mobilenet_v3'], help='Model to benchmark')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of inference runs')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--save', action='store_true',
                        help='Save results to benchmark_results.json')
    args = parser.parse_args()

    results = compare_cpu_gpu(args.image, args.model,
                              args.runs, args.batch_size)

    if args.save:
        os.makedirs("logs", exist_ok=True)
        with open("logs/benchmark_results.json", "w") as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'results': results
            }, f, indent=2)
        print("\n💾 Results saved to logs/benchmark_results.json")


if __name__ == "__main__":
    main()
