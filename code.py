import os
import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


train_path = "data/classification-of-pet-facial-expression/train/train"
test_path  = "data/classification-of-pet-facial-expression/test/test"

image_size = 48 

# print(os.listdir(train_path))

X_train = []
y_train = []

for label in os.listdir(train_path):
    class_path = os.path.join(train_path, label)
    print(class_path)

    # Skip non-folders like .DS_Store
    # if not os.path.isdir(class_path):
    #     continue

    for img_name in os.listdir(class_path):
        # print(img_name)
        img_path = os.path.join(class_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        # Convert to grayscale (reduces dimensionality)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize to fixed size
        img = cv2.resize(img, (image_size, image_size))

        # Normalize pixels to [0,1]
        img = img / 255.0

        # Flatten into 1D vector
        img = img.flatten()

        X_train.append(img)
        y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

print("Train shape:", X_train.shape)


le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)


#Splitting and  training
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_train_encoded
)


scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_val = scaler.transform(X_val)

from sklearn.decomposition import PCA

pca = PCA(n_components=200)

X_tr = pca.fit_transform(X_tr)
X_val = pca.transform(X_val)


model = LogisticRegression(
    l1_ratio=0,
    
    solver="lbfgs",
    max_iter=1000
)

model.fit(X_tr, y_tr)


y_val_pred = model.predict(X_val)

print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

X_test = []
test_ids = []

for img_name in os.listdir(test_path):
    img_path = os.path.join(test_path, img_name)

    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (image_size, image_size))
    img = img / 255.0
    img = img.flatten()

    X_test.append(img)
    test_ids.append(img_name)

X_test = np.array(X_test)
# Apply SAME scaler
X_test = scaler.transform(X_test)
X_test = pca.transform(X_test)




y_test_pred_encoded = model.predict(X_test)

# Convert numbers back to original labels
y_test_pred = le.inverse_transform(y_test_pred_encoded)

submission = pd.DataFrame({
    "id": test_ids,
    "label": y_test_pred
})

submission.to_csv("submission9.csv"+"", index=False)

print("Submission file created.")

