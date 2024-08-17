from torch.utils.data import Dataset

class SignatureDataset(Dataset):
    def __init__(self, authentic_signatures, forged_signatures, transform=None):
        self.authentic_signatures = authentic_signatures
        self.forged_signatures = forged_signatures
        self.transform = transform
        self.pairs, self.labels = self.create_pairs()

    def create_pairs(self):
        pairs = []
        labels = []
        
        # Positive pairs
        for i in range(len(self.authentic_signatures)):
            for j in range(i+1, len(self.authentic_signatures)):
                pairs.append([self.authentic_signatures[i], self.authentic_signatures[j]])
                labels.append(0)  # Same class
        
        # Negative pairs
        for i in range(len(self.authentic_signatures)):
            for j in range(len(self.forged_signatures)):
                pairs.append([self.authentic_signatures[i], self.forged_signatures[j]])
                labels.append(1)  # Different classes
        
        return pairs, labels
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        image1 = self.pairs[idx][0]
        image2 = self.pairs[idx][1]
        label = self.labels[idx]
        
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        
        return image1, image2, torch.tensor(label, dtype=torch.float32)
