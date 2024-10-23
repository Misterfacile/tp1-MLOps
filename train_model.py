import pandas as pd
import torch
import joblib

from sklearn.linear_model import LinearRegression
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

def build_model():
    df = pd.read_csv("./houses.csv")
    X = df[['size', 'nb_rooms', 'garden']]
    y = df['price']
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, "regression.joblib")
    
# build_model()

## BUILDING WITH BERT MODEL
def build_bert_model():
    df = pd.read_csv("./houses.csv")
    df['text'] = df.apply(lambda x: f"This house is {x['size']} square meters, has {x['nb_rooms']} bedrooms, and {'has' if x['garden'] == 1 else 'does not have'} a garden.", axis=1)

    X = df['text'].tolist()
    y = df['price'].tolist()
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    X_tokenized = tokenizer(X, padding=True, truncation=True, return_tensors='pt', max_length=128)
    y_tensor = torch.tensor(y).float()
    
    X_train, X_test, y_train, y_test = train_test_split(X_tokenized['input_ids'], y_tensor, test_size=0.1, random_state=42)
    
    class HousePriceDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {
                'input_ids': self.encodings[idx],
                'attention_mask': self.encodings[idx],
                'labels': self.labels[idx]
            }
            return item

        def __len__(self):
            return len(self.labels)

        
    train_dataset = HousePriceDataset(X_train, y_train)
    test_dataset = HousePriceDataset(X_test, y_test)
    
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=1)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,  # Start with fewer epochs
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        report_to=None,  # Disable W&B and any reporting
        learning_rate=5e-5,  # Adjust learning rate
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    
    model.save_pretrained('./bert_regression_model')  # Saves the entire model including configuration
    tokenizer.save_pretrained('./bert_regression_model')  # Saves the tokenizer