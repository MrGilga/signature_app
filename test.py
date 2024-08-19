# test.py
from PIL import Image
import torch
from torchvision import transforms
from model import SiameseNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# path of 2 images to test
realimage1 = 'Dataset/dataset1/real/00100001.png'
realimage2 = 'Dataset/dataset1/real/00101001.png'  
realimage3 = 'Dataset/dataset1/real/00101001.png'
forgeimage1 = 'Dataset/dataset1/forge/02100001.png'
forgeimage2 = 'Dataset/dataset1/forge/02101001.png'

# Le stesse trasformazioni utilizzate durante l'addestramento
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Riduci la dimensione dell'immagine
    transforms.ToTensor(),
])

threshold = 1.0

def eval(image1, image2):
    image1 = Image.open(image1).convert('L')
    image2 = Image.open(image2).convert('L')

    image1 = transform(image1).unsqueeze(0).to(device)
    image2 = transform(image2).unsqueeze(0).to(device)

    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    with torch.no_grad():
        output1, output2 = model(image1, image2)
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)

    return euclidean_distance.item() < threshold

assert eval(realimage1, realimage2) == True
assert eval(realimage1, realimage3) == True
assert eval(realimage1, forgeimage1) == False
assert eval(real
