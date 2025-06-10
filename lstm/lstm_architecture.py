import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self,input_size = 258,hidden_size = 160,output_size = 9):
        super(LSTMModel,self).__init__()
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers = 1,batch_first = True)
        self.dropout = nn.Dropout(0.6)
        self.fc1 = nn.Linear(hidden_size,hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,output_size)
    
    def forward(self,x):
        x, _ = self.lstm(x)
        x = torch.max(x,dim = 1).values
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x 


# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, dropout_rate=0.5):
#         super(LSTMModel, self).__init__()
        
#         # First LSTM layer with return_sequences=True (return all hidden states)
#         self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
#         self.bn1 = nn.BatchNorm1d(hidden_size1)
#         self.dropout1 = nn.Dropout(dropout_rate)
        
#         # Second LSTM layer returns only the last output
#         self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
#         self.bn2 = nn.BatchNorm1d(hidden_size2)
#         self.dropout2 = nn.Dropout(dropout_rate)
        
#         # Dense layer with ReLU
#         self.dense = nn.Linear(hidden_size2, 32)
#         self.bn3 = nn.BatchNorm1d(32)
#         self.relu = nn.ReLU()
        
#         # Output layer
#         self.output = nn.Linear(32, num_classes)
#         self.softmax = nn.Softmax(dim=1)
        
#     def forward(self, x):
#         # First LSTM layer (return all sequences)
#         lstm1_out, _ = self.lstm1(x)
#         # Apply batch norm across the timesteps for each feature
#         # We need to reshape for batch norm to work on LSTM output
#         batch_size, seq_len, features = lstm1_out.size()
#         lstm1_out_reshaped = lstm1_out.contiguous().view(-1, features)
#         lstm1_out_bn = self.bn1(lstm1_out_reshaped)
#         lstm1_out = lstm1_out_bn.view(batch_size, seq_len, features)
#         lstm1_out = self.dropout1(lstm1_out)
        
#         # Second LSTM layer (return only last output)
#         lstm2_out, _ = self.lstm2(lstm1_out)
#         # Get the last time step output
#         lstm2_out = lstm2_out[:, -1, :]
#         lstm2_out = self.bn2(lstm2_out)
#         lstm2_out = self.dropout2(lstm2_out)
        
#         # Dense layer
#         dense_out = self.dense(lstm2_out)
#         dense_out = self.bn3(dense_out)
#         dense_out = self.relu(dense_out)
        
#         # Output layer
#         output = self.output(dense_out)
        
#         # During training, we'll use CrossEntropyLoss which applies softmax internally
#         # so we don't apply softmax here for training
#         # For inference, we can apply softmax: return self.softmax(output)
#         return output