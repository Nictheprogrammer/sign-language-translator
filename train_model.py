import csv
import pickle
import random
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

print("Loading dataset...")

# Read the CSV file
rows = []
with open('dataset.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header row
    for row in reader:
        rows.append(row)

# Shuffle the data
random.shuffle(rows)

# Split into features (X) and labels (y)
X = []
y = []
for row in rows:
    features = [float(val) for val in row[:-1]]
    label = row[-1]
    X.append(features)
    y.append(label)

print(f"Dataset loaded: {len(X)} samples, {len(X[0])} features each")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the data
# This is new -- explained below
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Training on {len(X_train)} samples...")
print(f"Testing on {len(X_test)} samples...")
print("This may take a minute...")

# Create and train the neural network
model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),  # 3 layers with 256, 128, 64 neurons
    activation='relu',                   # activation function
    max_iter=1000,                       # maximum training rounds
    random_state=42,
    verbose=True                         # prints progress while training
)
model.fit(X_train, y_train)

print("\nTraining complete!")

# Test the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Show accuracy per letter
print("\nAccuracy per letter:")
print(classification_report(y_test, predictions))

# Save BOTH the model and the scaler
# We need both because we must scale new data the same way
with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)

print("\nModel saved to model.pkl!")