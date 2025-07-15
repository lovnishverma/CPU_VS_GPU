import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
from PIL import Image

def run_inference(image_path="example.jpg", log_result=True):
    print("Loading model...")
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()

    transform = weights.transforms()

    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)

    print("Running inference...")
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    end_time = time.time()

    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.4f} seconds")

    # 🔍 Get the predicted class
    predicted_class_idx = output.argmax().item()
    class_name = weights.meta["categories"][predicted_class_idx]
    print(f"Predicted object: {class_name}")

    if log_result:
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        with open("inference_log.txt", "a") as log_file:
            log_file.write(f"{device_name} - Inference time: {inference_time:.4f} sec - Detected: {class_name}\n")

if __name__ == "__main__":
    run_inference("example.jpg")
