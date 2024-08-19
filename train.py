#%%
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#%%
# Percorsi delle cartelle
authentic_dir = 'Dataset/dataset1/real' 
forged_dir = 'Dataset/dataset1/forge'      

transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),
])

#%%
class SignatureDataset(Dataset):
    def __init__(self, authentic_dir, forged_dir, transform=None):
        n_ppl = 12
        self.images = [{"real":[], 'forge': []} for _ in range(n_ppl)]

        for img in os.listdir(authentic_dir):
            realid = int(img[-7:-4])
            self.images[realid-1]['real'].append(os.path.join(authentic_dir, img))
        for img in os.listdir(forged_dir):
            realid = int(img[-7:-4])
            self.images[realid-1]['forge'].append(os.path.join(forged_dir, img))

        self.transform = transform
        self.pairs, self.labels = self.create_pairs()
    
    def create_pairs(self):
        n_per_person = 5
        pairs = []
        labels = []

        for person in self.images:
            for i, real1 in enumerate(person['real']):
                for real2 in person['real'][i+1:]:
                    pairs.append([real1, real2])
                    labels.append(0)
                for forge in person['forge']:
                    pairs.append([real1, forge])
                    labels.append(1)
                    

        return pairs, labels
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        image1_path = self.pairs[idx][0]
        image2_path = self.pairs[idx][1]
        label = self.labels[idx]
        
        image1 = Image.open(image1_path).convert('L')  
        image2 = Image.open(image2_path).convert('L')
        
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        
        return image1, image2, torch.tensor(label, dtype=torch.float32)
#%%
dataset = SignatureDataset(authentic_dir, forged_dir, transform=transform)

dataloader = DataLoader(dataset, shuffle=True, batch_size=32)

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.optim as optim
from model import SiameseNetwork, ContrastiveLoss
model = SiameseNetwork().to(device)

criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# %%
num_epochs = 5 

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, data in enumerate(dataloader, 0):
        img1, img2, label = data
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        
        optimizer.zero_grad() 

        output1, output2 = model(img1, img2)
        
        loss = criterion(output1, output2, label)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')