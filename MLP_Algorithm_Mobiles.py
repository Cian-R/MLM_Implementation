import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

seed = 4211
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

csv_file = 'data/data.csv'
used_columns = ["battery_power", "int_memory", "n_cores", "ram", "price_range"]

raw_data = pd.read_csv(csv_file)
data_file = raw_data[used_columns]

scaler = MinMaxScaler()

data_X = data_file.iloc[:, :-1]
data_X = pd.DataFrame(scaler.fit_transform(data_X), columns=data_X.columns)

data_Y = data_file.iloc[:, -1]
data_Y = data_Y.replace({0: 0, 1: 0, 2: 1, 3: 1}) # Price range column now contains binary indicator of whether phone is high price or low price.

print(data_X)
print(data_Y)

data_X = data_X.to_numpy()
data_Y = data_Y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=42)
print("\n\n\n\n")
print(type(X_train))
print(X_train)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 8)  # Input layer to first hidden layer
        self.relu1 = nn.ReLU()      # First ReLU activation function
        self.fc2 = nn.Linear(8, 8)  # First hidden layer to second hidden layer
        self.relu2 = nn.ReLU()      # Second ReLU activation function
        self.fc3 = nn.Linear(8, 2)  # Second hidden layer to output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

model = MLP()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

train_losses = []  # For storing training loss over epochs
train_accuracies = []  # For storing training accuracy over epochs
val_accuracies = []  # For storing validation accuracy over epochs

num_epochs = 3000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculate training accuracy for this epoch
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == y_train).sum().item()
    accuracy = correct / y_train.size(0)
    train_accuracies.append(accuracy)

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')



plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training Accuracy Over Epochs')

plt.tight_layout()
plt.show()

model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)

cm = confusion_matrix(y_test, predicted)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


correct = (predicted == y_test).sum().item()
total = y_test.size(0)
accuracy = correct / total
print(f'Accuracy on Test Data: {accuracy * 100:.2f}%')
