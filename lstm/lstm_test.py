import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from avg_tweet_lstm import TweetAVGChronologicalLSTM

class TestDataset(Dataset):
    def __init__(self, data_frame, sequence_length=5):
        """
        Similar to MatchDataset but handles edge cases for test data
        """
        self.sequence_length = sequence_length
        self.sequences = []
        self.ids = []  # Store original IDs
        
        # Group by match
        for match_id, match_data in data_frame.groupby('MatchID'):
            # Sort by PeriodID and ID to ensure chronological order
            match_data = match_data.sort_values(['PeriodID'])
            
            # Get embeddings
            embeddings = match_data.iloc[:, 3:].values  # GloVe features start at column 4
            original_ids = match_data['ID'].values
            
            # Handle the start of the match where we don't have enough history
            # Pad with zeros at the beginning
            padding = np.zeros((sequence_length - 1, embeddings.shape[1]))
            padded_embeddings = np.vstack([padding, embeddings])
            
            # Create sequences for every minute, including the first ones
            for i in range(len(embeddings)):
                start_idx = i
                end_idx = i + sequence_length
                sequence = padded_embeddings[start_idx:end_idx]
                self.sequences.append(sequence)
                self.ids.append(original_ids[i])
        
        self.sequences = torch.FloatTensor(np.array(self.sequences))
        self.ids = np.array(self.ids)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.ids[idx]

def predict_and_export(model, test_file_path, output_file_path, sequence_length=5):
    """
    Make predictions on test data and export to CSV
    """
    # Load the test data
    test_df = pd.read_csv(test_file_path)
    test_dataset = TestDataset(test_df, sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Set model to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Make predictions
    all_preds = []
    all_ids = []
    
    with torch.no_grad():
        for data, ids in test_loader:
            data = data.to(device)
            outputs = model(data)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_ids.extend(ids)
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'ID': all_ids,
        'EventType': all_preds
    })
    
    # Export to CSV
    output_df.to_csv(output_file_path, index=False)
    
    return output_df

# Usage example:
def main():
    # Load your trained model
    model = TweetAVGChronologicalLSTM(
        input_dim=200,
        hidden_dim=32,
        num_classes=2
    )
    
    # Load the saved model weights
    model.load_state_dict(torch.load('best_model.pt'))
    
    # Make predictions and export
    predictions = predict_and_export(
        model=model,
        test_file_path='period_features_test_glove.csv',
        output_file_path='lstm_like_paper.csv'
    )
    
    print("Predictions have been saved to predictions.csv")
    # Print some statistics
    print(f"\nPrediction statistics:")
    print(f"Total predictions: {len(predictions)}")
    print(f"Number of events detected: {sum(predictions['EventType'])}")
    print(f"Percentage of events: {(sum(predictions['EventType'])/len(predictions))*100:.2f}%")

if __name__ == "__main__":
    main()