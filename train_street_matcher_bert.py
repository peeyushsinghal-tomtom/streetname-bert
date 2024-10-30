import torch
from torch.utils.data import Dataset, DataLoader
from street_matcher_bert import StreetMatcherBERT
import torch.optim as optim

class StreetPairDataset(Dataset):
    def __init__(self, street_pairs, labels):
        """
        Args:
            street_pairs: List of tuples (street1, street2)
            labels: List of tuples (confidence, is_match)
        """
        self.street_pairs = street_pairs
        self.labels = labels
        
    def __len__(self):
        return len(self.street_pairs)
    
    def __getitem__(self, idx):
        return self.street_pairs[idx], self.labels[idx]

def train_model(model, train_loader, num_epochs=10, learning_rate=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    confidence_criterion = torch.nn.MSELoss()
    match_criterion = torch.nn.BCELoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (street_pairs, labels) in enumerate(train_loader):
            street1, street2 = zip(*street_pairs)
            confidence_labels, match_labels = zip(*labels)
            
            confidence_labels = torch.tensor(confidence_labels, dtype=torch.float).to(device)
            match_labels = torch.tensor(match_labels, dtype=torch.float).to(device)
            
            optimizer.zero_grad()
            
            confidence, match_pred = model(list(street1), list(street2))
            
            # Calculate combined loss
            confidence_loss = confidence_criterion(confidence, confidence_labels)
            match_loss = match_criterion(match_pred.float(), match_labels)
            loss = confidence_loss + match_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')

# Example training usage
if __name__ == "__main__":
    # Create sample training data
    street_pairs = [
        ("Main Street", "Main St"),
        ("Broadway Ave", "Broadway Avenue"),
        ("Fifth Street", "5th Street"),
        ("Hauptstraße", "Hauptstrasse"),
        ("Rue de la Paix", "Rue de la Paix")
    ]
    
    # Labels: (confidence, is_match)
    labels = [
        (0.9, 1),  # High confidence match
        (0.95, 1), # High confidence match
        (0.8, 1),  # Medium-high confidence match
        (0.85, 1), # High confidence match
        (1.0, 1),  # Perfect match
    ]
    
    dataset = StreetPairDataset(street_pairs, labels)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = StreetMatcherBERT()
    train_model(model, train_loader)
    
    # Save the trained model
    torch.save(model.state_dict(), 'street_matcher.pth')
    
    # Rest of the file remains the same 