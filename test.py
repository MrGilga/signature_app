#%%
from PIL import Image
import torch
from torchvision import transforms
from model import SiameseNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

realimage1 = 'real/00101001.png'
realimage2 = 'preprocessed_real/00100001.png' 
realimage3 = 'real/00104001.png'
forgeimage1 = 'preprocessed_forge/02100001.png'
forgeimage2 = 'preprocessed_forge/02103001.png'

transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Riduce image dimensions to 64x64    
    transforms.ToTensor(),
])

threshold = 1.0

def eval(image1, image2, return_distance=False):
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

    if return_distance:
        return euclidean_distance.item()
    return euclidean_distance.item() < threshold


print("Euclidean distance between realimage1 and realimage2:", eval(realimage1, realimage2, return_distance=True))
print("Euclidean distance between realimage1 and realimage3:", eval(realimage1, realimage3, return_distance=True))
print("Euclidean distance between realimage1 and forgeimage1:", eval(realimage1, forgeimage1, return_distance=True))
print("Euclidean distance between realimage1 and forgeimage2:", eval(realimage1, forgeimage2, return_distance=True))


# %%
