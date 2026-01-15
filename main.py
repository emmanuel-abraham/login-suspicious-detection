from src.data_preprocessing import preprocess_data
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.visualization import (
    plot_login_distribution,
    plot_normal_vs_suspicious,
    plot_confusion_matrix
)

# Path to dataset
DATA_PATH = "dataset/login_data.csv"

def main():
    # Step 1: Data preprocessing
    data, X, y = preprocess_data(DATA_PATH)

    # Step 2: Train machine learning model
    model, X_test, y_test = train_model(X, y)

    # Step 3: Evaluate model
    accuracy, matrix = evaluate_model(model, X_test, y_test)

    # Display results
    print("Model Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(matrix)

    # Step 4: Visualization
    plot_login_distribution(data)
    plot_normal_vs_suspicious(data)
    plot_confusion_matrix(matrix)

if __name__ == "__main__":
    main()
