import joblib
import torch
import pandas as pd
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertModel

def build_model():
    df = pd.read_csv("./houses.csv")
    X = df[['size', 'nb_rooms', 'garden']]
    y = df['price']
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, "regression.joblib")

class DistilBertForRegression(nn.Module):
    def __init__(self):
        super(DistilBertForRegression, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return self.regressor(outputs.last_hidden_state[:, 0, :])

def build_bert_model():
    df = pd.read_csv("./houses.csv")
    df['text'] = df.apply(lambda x: f"This house is {x['size']} square meters, has {x['nb_rooms']} bedrooms, and {'has' if x['garden'] == 1 else 'does not have'} a garden.", axis=1)

    X = df['text'].tolist()
    y = df['price'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y).flatten()
    print(y_scaled)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    X_tokenized = tokenizer(X, padding=True, truncation=True, return_tensors='pt', max_length=128)
    input_ids = X_tokenized['input_ids']
    attention_mask = X_tokenized['attention_mask']

    assert len(input_ids) == len(y_scaled), "Input and labels must have the same number of samples."
    X_train_ids, X_test_ids, y_train, y_test = train_test_split(input_ids, y_scaled, test_size=0.2, random_state=42)
    X_train_mask, X_test_mask = train_test_split(attention_mask, test_size=0.2, random_state=42)
    class HousePriceDataset(torch.utils.data.Dataset):
        def __init__(self, input_ids, attention_mask, labels):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.labels = labels

        def __getitem__(self, idx):
            item = {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
            }
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = HousePriceDataset(X_train_ids, X_train_mask, y_train)
    model = DistilBertForRegression()
    
    for param in model.bert.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    model.train()
    for epoch in range(5):
        for batch in train_dataset:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].unsqueeze(0)
            attention_mask = batch['attention_mask'].unsqueeze(0)
            labels = batch['labels'].unsqueeze(0)

            outputs = model(input_ids, attention_mask)
            loss = nn.MSELoss()(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), './bert_regression_model/model.pth')
    tokenizer.save_pretrained('./bert_regression_model/tokenizer/')