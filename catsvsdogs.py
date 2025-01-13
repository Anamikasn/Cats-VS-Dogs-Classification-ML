import os
import numpy as np
import cv2
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from skimage.feature import hog
from sklearn.metrics import classification_report, accuracy_score

# Define directories and categories
dir = 'C:\\Users\\Anamika S Nair\\Downloads\\test_set\\test_set'
categories = ['Cat', 'Dog']

# Function to extract HOG features
def extract_features(image):
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

# Prepare data
data = []
for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)
    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        pet_img = cv2.imread(imgpath, 0)
        try:
            pet_img = cv2.resize(pet_img, (100, 100))  # Resize to 100x100
            features = extract_features(pet_img)
            data.append([features, label])
        except Exception as e:
            pass

# Split data into features and labels
X = np.array([item[0] for item in data])  # Features
y = np.array([item[1] for item in data])  # Labels

# Count total images in the dataset
print(f"Total images in dataset: {len(data)}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Count training and testing images
print(f"Total training images: {len(X_train)}")
print(f"Total testing images: {len(X_test)}")

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)

# Use the best model found
model = grid.best_estimator_

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions) * 100
print(f"Model accuracy on test set: {accuracy:.2f}%")
print(classification_report(y_test, predictions, target_names=categories))

# Save the trained model
with open('data1.pickle', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as data1.pickle")