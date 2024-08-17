#%%
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#%%
# Percorsi delle cartelle
authentic_dir = 'Dataset/dataset1/real'  # Modifica con il percorso della cartella delle firme autentiche
forged_dir = 'Dataset/dataset1/forge'       # Modifica con il percorso della cartella delle firme false

# Trasformazione per convertire le immagini in tensori
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Riduci la dimensione dell'immagine
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

    model.eval()

    with torch.no_grad():
        output1, output2 = model(image1, image2)
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)

    return euclidean_distance.item() < threshold

assert eval(realimage1, realimage2) == True
assert eval(realimage1, realimage3) == True
assert eval(realimage1, forgeimage1) == False
assert eval(realimage1, forgeimage2) == True