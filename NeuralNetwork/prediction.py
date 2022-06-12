import torch
import torchvision.transforms as transforms

from PIL import Image
from NeuralNetwork import NeuralNetwork

def getPrediction(img_path):
    
    neural_net = NeuralNetwork.NeuralNetwork()
    PATH = './trained_nn.pth'
    img = Image.open(img_path)
    transform_tensor = transforms.ToTensor()(img).unsqueeze_(0)
    classes = ['glass', 'metal', 'paper', 'plastic']
    neural_net.load_state_dict(torch.load(PATH))
    neural_net.eval()
    outputs = neural_net(transform_tensor)

    return classes[torch.max(outputs, 1)[1]]