import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score

class MatchDataset(Dataset):
    def __init__(self, data_frame, sequence_length=5):
        """
        data_frame: DataFrame with columns [MatchID, PeriodID, ID, EventType, embedding features...]
        sequence_length: Number of minutes to consider as context
        """
        self.sequence_length = sequence_length
        self.sequences = []
        self.labels = []
        
        # Group by match
        for match_id, match_data in data_frame.groupby('MatchID'):
            # Sort by PeriodID and ID to ensure chronological order
            match_data = match_data.sort_values(['PeriodID'])
            
            # Get embeddings and labels
            embeddings = match_data.iloc[:, 4:].values  # GloVe features start at column 4
            event_labels = match_data['EventType'].values
            
            # Create sequences
            for i in range(len(embeddings) - sequence_length + 1):
                self.sequences.append(embeddings[i:i + sequence_length])
                # Label is the event type of the last minute in the sequence
                self.labels.append(event_labels[i + sequence_length - 1])
        
        self.sequences = torch.FloatTensor(np.array(self.sequences))
        self.labels = torch.LongTensor(self.labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class TweetAVGChronologicalLSTM(nn.Module):
    def __init__(self, input_dim=200, hidden_dim=32, num_classes=2):  # hidden_dim=32 as per config
        super(TweetAVGChronologicalLSTM, self).__init__()
        
        # Batch normalization (batch_norm = True)
        self.batch_norm = nn.BatchNorm1d(input_dim)
        
        # Single layer LSTM (num_lstm_layers = 1)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,  # as per config
            batch_first=True
        )
        
        # Dropouts matching their config
        self.dropout_embedding = nn.Dropout(0.1)  # dropout_embedding = 0.1
        self.dropout_lstm_out = nn.Dropout(0.1)   # dropout_lstm2_output = 0.1
        
        # Classification head
        self.hidden2tag = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # Input is already averaged embeddings
        batch_size, seq_len, feat_dim = x.size()
        
        # Apply embedding dropout
        x = self.dropout_embedding(x)
        
        # Batch norm
        if seq_len > 1:
            x = x.view(-1, feat_dim)
            x = self.batch_norm(x)
            x = x.view(batch_size, seq_len, feat_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # LSTM output dropout
        lstm_out = self.dropout_lstm_out(lstm_out)
        
        # Get predictions for each timestep
        tag_space = self.hidden2tag(lstm_out[:, -1, :])
        
        return tag_space

def train_model(model, train_loader, val_loader, num_epochs=150, learning_rate=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_f1 = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = output.argmax(dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(target.cpu().numpy())
        
        train_f1 = f1_score(train_labels, train_preds, average='micro')
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                preds = output.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(target.cpu().numpy())
        
        val_f1 = f1_score(val_labels, val_preds, average='micro')
        
        print(f'Epoch {epoch+1}:')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}, F1: {train_f1:.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}, F1: {val_f1:.4f}')
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pt')

def prepare_data(csv_path, sequence_length=5, val_ratio=0.2):
    """Prepare the data from CSV file"""
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Split based on matches
    unique_matches = df['MatchID'].unique()
    np.random.shuffle(unique_matches)
    split_idx = int(len(unique_matches) * (1 - val_ratio))
    
    train_matches = unique_matches[:split_idx]
    val_matches = unique_matches[split_idx:]
    
    train_df = df[df['MatchID'].isin(train_matches)]
    val_df = df[df['MatchID'].isin(val_matches)]
    
    # Create datasets
    train_dataset = MatchDataset(train_df, sequence_length)
    val_dataset = MatchDataset(val_df, sequence_length)
    
    return train_dataset, val_dataset

# Usage example:
def main():
    sequence_length = 5  # Look at 5 minutes of context
    batch_size = 32
    
    # Load and prepare data
    train_dataset, val_dataset = prepare_data('period_features_glove.csv', sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    model = TweetAVGChronologicalLSTM(
        input_dim=200,  # Your GloVe embedding dimension
        hidden_dim=32,
        num_classes=2  # Binary classification
    )
    
    # Train model
    train_model(model, train_loader, val_loader)

if __name__ == "__main__":
    main()