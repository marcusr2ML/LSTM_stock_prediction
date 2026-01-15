import torch
from torch.nn import LSTM
from torch import nn

class lstmTimeSeries(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, 
                 num_layers=2, output_size=1,
                 bias=True, batch_first=True, dropout=0, 
                 bidirectional=True, freeze=False):
        super().__init__()
        # LSTM
        self.time_series = LSTM(input_size, hidden_size, num_layers, bias,
                                batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        # Fully connected output layer
        hidden_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(p=0.3),       # 30% dropout
            nn.Linear(hidden_size//2, output_size)           
        )
        if freeze:
            self._freeze()
        
        # Device handling
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def forward(self, x):
        out, (h_n, c_n) = self.time_series(x)
        # Use last timestep's output
        out = self.fc(out[:, -1, :])
        return out

    def _freeze(self):        
        for p in self.time_series.parameters():
            p.requires_grad = False
            
    # --- Save model ---
    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'device': self.device
        }, path)
        print(f"Model saved to {path}")

    # --- Load model ---
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.to(self.device)
        print(f"Model loaded from {path}")
