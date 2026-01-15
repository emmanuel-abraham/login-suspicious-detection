import matplotlib.pyplot as plt

def plot_login_distribution(data):
    """
    Displays login frequency distribution by hour.
    """
    plt.hist(data['login_hour'], bins=24)
    plt.xlabel("Login Hour")
    plt.ylabel("Number of Login Attempts")
    plt.title("Login Frequency Distribution by Hour")
    plt.show()

def plot_normal_vs_suspicious(data):
    """
    Displays normal vs suspicious login activity distribution.
    """
    labels = ['Normal', 'Suspicious']
    counts = data['login_success'].value_counts().sort_index()
    
    plt.pie(counts, labels=labels, autopct='%1.1f%%')
    plt.title("Normal vs Suspicious Login Activity Distribution")
    plt.show()

def plot_confusion_matrix(matrix):
    """
    Displays confusion matrix.
    """
    plt.imshow(matrix)
    plt.title("Confusion Matrix for Login Classification")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.show()
