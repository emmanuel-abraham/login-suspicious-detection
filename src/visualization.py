import matplotlib.pyplot as plt

def plot_login_distribution(data):
    plt.hist(data['login_hour'], bins=24)
    plt.xlabel("Login Hour")
    plt.ylabel("Number of Login Attempts")
    plt.title("Login Frequency Distribution by Hour")
    plt.show()

def plot_confusion_matrix(matrix):
    plt.imshow(matrix)
    plt.title("Confusion Matrix for Login Classification")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.show()
