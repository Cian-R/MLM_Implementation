import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from mlxtend.plotting import plot_decision_regions

seed = 12443
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

iris = load_iris()
X = iris.data[:100, :2]  # Select the first two features
y = iris.target[:100]    # Labels (0 for setosa, 1 for versicolor)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)


# Define a custom plotting function
def plot_decision_boundary(X, y, model, ax=None):
    if ax is None:
        ax = plt.gca()

    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    Z = model(grid_points).detach().numpy()
    
    if len(Z[0]) == 2:
        # For binary classification, create a single decision boundary line
        decision_boundary = Z[:, 1] - Z[:, 0]  # Assuming class 1 is the positive class
        decision_boundary = decision_boundary.reshape(xx.shape)
        ax.contour(xx, yy, decision_boundary, colors='k', levels=[0], linestyles='solid')
    else:
        # For multi-class classification, create separate decision boundary lines for each class
        for i in range(len(Z[0])):
            class_region = Z[:, i].reshape(xx.shape)
            ax.contour(xx, yy, class_region, colors='k', levels=[0], linestyles='solid')
    
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')



class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # Input layer to first hidden layer
        self.relu1 = nn.ReLU()      # First ReLU activation function
        self.fc2 = nn.Linear(4, 4)  # First hidden layer to second hidden layer
        self.relu2 = nn.ReLU()      # Second ReLU activation function
        self.fc3 = nn.Linear(4, 2)  # Second hidden layer to output layer

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

num_epochs = 1000
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

# Usage
plot_decision_boundary(X_train, y_train, model)
plt.title('Decision Boundary')
plt.show()


correct = (predicted == y_test).sum().item()
total = y_test.size(0)
accuracy = correct / total
print(f'Accuracy on Test Data: {accuracy * 100:.2f}%')
