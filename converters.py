import torch
from torchvision.transforms import v2

def image_to_tensor(image):
  tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize((0.5), (0.5))])(image)
  return tensor.unsqueeze(0)

def tensor_to_image(tensor):
  tensor = tensor.squeeze()
  return v2.ToPILImage()(tensor)

def turn_to_grayscale(image):
  return v2.Grayscale()(image)
  
