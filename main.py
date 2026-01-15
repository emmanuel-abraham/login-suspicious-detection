from src.data_preprocessing import preprocess_data
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.visualization import plot_login_distribution, plot_confusion_matrix

DATA_PATH = "dataset/login_data.csv"

data, X, y = preprocess_data(DATA_PATH)
model, X_test, y_test = train_model(X, y)
accuracy, matrix = evaluate_model(model, X_test, y_test)

print("Model Accuracy:", accuracy)
print("Confusion Matrix:")
print(matrix)

plot_login_distribution(data)
plot_confusion_matrix(matrix)
