import os
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
from PIL import Image


def run_inference(
    image_path="example.jpg",
    log_result=True,
    log_file_path="inference_log.txt",
    device=None
):
    # Device setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    # Performance optimization for CUDA
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    # Load model
    print("Loading model...")
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()
    model.to(device)

    # Use half precision on CUDA
    use_half = device == "cuda"
    if use_half:
        model = model.half()

    # Image preprocessing
    transform = weights.transforms()
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    input_tensor = transform(img).unsqueeze(0).to(device)
    if use_half:
        input_tensor = input_tensor.half()

    # Warm-up pass for CUDA to reduce overhead
    with torch.no_grad():
        _ = model(input_tensor)

    # Timed inference
    print("Running inference...")
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    end_time = time.time()

    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.4f} seconds")

    # Predicted class
    predicted_class_idx = output.argmax().item()
    class_name = weights.meta["categories"][predicted_class_idx]
    print(f"Predicted object: {class_name}")

    # Logging results
    if log_result:
        device_name = torch.cuda.get_device_name(
            0) if device == "cuda" else "CPU"
        os.makedirs(os.path.dirname(log_file_path),
                    exist_ok=True) if os.path.dirname(log_file_path) else None
        with open(log_file_path, "a") as log_file:
            log_file.write(
                f"{device_name} - Inference time: {inference_time:.4f} sec - Detected: {class_name}\n"
            )


if __name__ == "__main__":
    run_inference("example.jpg")
