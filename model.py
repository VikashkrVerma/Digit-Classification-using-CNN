import numpy as np
import random
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.datasets import mnist
import itertools

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the dataset
X_train, X_test = X_train / 255.0, X_test / 255.0

# Print shapes of the datasets
print("Training dataset shape:", X_train.shape)
print("Testing dataset shape:", X_test.shape)

# Display a random image from the dataset
random_index = random.randint(0, X_train.shape[0]-1)  # Choose a random index
random_image = X_train[random_index].reshape(28, 28)   # Reshape the image to 28x28
plt.imshow(random_image, cmap='gray')                 # Display the image in grayscale
plt.axis('off')                                        # Turn off axis
plt.title(f"Label: {y_train[random_index]}")          # Set the title as the label
plt.show()

# Define baseline CNN model
def baseline_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Define improved CNN model
def improved_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),        
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Function to plot training history
def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy / Loss')
    plt.legend()
    plt.show()

# Train model function
def train_model(model, X_train, y_train):
    history = model.fit(X_train, y_train, epochs=10, verbose=1)
    return history

# Evaluate model function
def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("Test accuracy:", test_acc)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Function for error analysis
def error_analysis(model1, model2, X_test, y_test):
    preds_baseline = np.argmax(model1.predict(X_test), axis=1)
    preds_improved = np.argmax(model2.predict(X_test), axis=1)

    incorrect_baseline = np.where(preds_baseline != y_test)[0]
    incorrect_improved = np.where(preds_improved != y_test)[0]

    print("Examples where the improved model classified correctly whereas the baseline did not:")
    for idx in incorrect_baseline[:5]:
        print("Baseline misclassified as:", preds_baseline[idx], "Correct label:", y_test[idx])

    print("\nExamples where the baseline classified correctly whereas the improved model did not:")
    for idx in incorrect_improved[:5]:
        print("Improved model misclassified as:", preds_improved[idx], "Correct label:", y_test[idx])

# Load baseline model
baseline = baseline_model()

# Load improved model
improved = improved_model()

# Compile both models
baseline.compile(optimizer='sgd',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

improved.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
# Train baseline model
history_baseline = train_model(baseline, X_train, y_train)

# Train improved model
history_improved = train_model(improved, X_train, y_train)

# Plot training history for baseline model
plot_training_history(history_baseline)

# Plot training history for improved model
plot_training_history(history_improved)

# Evaluate baseline model
evaluate_model(baseline, X_test, y_test)

# Evaluate improved model
evaluate_model(improved, X_test, y_test)

# Generate confusion matrix for baseline model
baseline_preds = np.argmax(baseline.predict(X_test), axis=1)
baseline_cm = confusion_matrix(y_test, baseline_preds)
plot_confusion_matrix(baseline_cm, classes=np.arange(10))

# Generate confusion matrix for improved model
improved_preds = np.argmax(improved.predict(X_test), axis=1)
improved_cm = confusion_matrix(y_test, improved_preds)
plot_confusion_matrix(improved_cm, classes=np.arange(10))

# Error analysis
error_analysis(baseline, improved, X_test, y_test)
