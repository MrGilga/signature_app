#%%
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
#%%
# Percorsi delle cartelle
authentic_dir = 'preprocessed_forge'  # Modifica con il percorso della cartella delle firme autentiche
forged_dir = 'preprocessed_real'       # Modifica con il percorso della cartella delle firme false

# Trasformazione per convertire le immagini in tensori
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Riduci la dimensione dell'immagine
    transforms.ToTensor(),
])

#%%
class SignatureDataset(Dataset):
    def __init__(self, authentic_dir, forged_dir, transform=None):
        self.authentic_images = [os.path.join(authentic_dir, img) for img in os.listdir(authentic_dir)]
        self.forged_images = [os.path.join(forged_dir, img) for img in os.listdir(forged_dir)]
        self.transform = transform
        self.pairs, self.labels = self.create_pairs()
    
    def create_pairs(self):
        pairs = []
        labels = []
        
        # Coppie positive
        for i in range(len(self.authentic_images)):
            for j in range(i+1, len(self.authentic_images)):
                pairs.append([self.authentic_images[i], self.authentic_images[j]])
                labels.append(0)  # Stessa classe
        
        # Coppie negative
        for i in range(len(self.authentic_images)):
            for j in range(len(self.forged_images)):
                pairs.append([self.authentic_images[i], self.forged_images[j]])
                labels.append(1)  # Diverse classi
        
        return pairs, labels
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        image1_path = self.pairs[idx][0]
        image2_path = self.pairs[idx][1]
        label = self.labels[idx]
        
        image1 = Image.open(image1_path).convert('L')  # Converti in scala di grigi se non lo è già
        image2 = Image.open(image2_path).convert('L')
        
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        
        return image1, image2, torch.tensor(label, dtype=torch.float32)
#%%
# Creazione del dataset
dataset = SignatureDataset(authentic_dir, forged_dir, transform=transform)

# Creazione del DataLoader
dataloader = DataLoader(dataset, shuffle=True, batch_size=32)

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.optim as optim
from model import SiameseNetwork, ContrastiveLoss
# Inizializza il modello
model = SiameseNetwork().to(device)

# Definisci la funzione di perdita e l'ottimizzatore
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# %%
num_epochs = 5  # Puoi regolare questo numero in base alle tue esigenze

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, data in enumerate(dataloader, 0):
        img1, img2, label = data
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        
        optimizer.zero_grad()  # Resetta i gradienti

        # Forward pass: calcola le feature per entrambe le immagini
        output1, output2 = model(img1, img2)
        
        # Calcola la perdita
        loss = criterion(output1, output2, label)
        
        # Backward pass e ottimizzazione
        loss.backward()
        optimizer.step()
        
        # Accumula la perdita
        running_loss += loss.item()
    
    # Stampa la perdita media per l'epoca corrente
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')
# %%
