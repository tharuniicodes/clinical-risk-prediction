from src.preprocess import preprocess_data

X_train, X_test, y_train, y_test, features = preprocess_data("data/heart.csv")

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Features:", features)
