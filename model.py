import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # Una CNN molto semplice con soli 2 livelli
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        
        # Livelli fully connected semplificati
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),  # Dimensione del tensore ridotta
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
    
    def forward_once(self, x):
        # Forward pass attraverso la CNN
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)  # Flatten
        output = self.fc1(output)
        return output
    
    def forward(self, input1, input2):
        # Passa entrambe le immagini
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


# Loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
