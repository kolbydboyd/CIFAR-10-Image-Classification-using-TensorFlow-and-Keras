# CIFAR-10 Image Classification using TensorFlow and Keras
This project demonstrates image classification on the CIFAR-10 dataset using TensorFlow and Keras. It includes code for data preprocessing, model building, training, evaluation, and visualization of results.

## Installation
#### We will be using TensorFlow, Keras, Numpy, and MatPlotLib to accomplish our goal.
#### You can install these required libraries using pip:

```bash
!pip install tensorflow
!pip install keras
!pip install numpy
!pip install matplotlib
```

#### Next, we need to import everything we will need for successful training.

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from collections import Counter
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
```

#### We chose to use the CIFAR-10 dataset.

```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

#### After loading the dataset, let's visualize a few random images from your dataset to understand its content and overall quality.

```python
# Display sample images
plt.figure(figsize=(10, 5))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_train[i])
plt.show()
```

#### Then, let's check the shape of the data to confirm the number of images and their dimensions.

```python
print('Training data shape:', x_train.shape)
print('Training labels shape:', y_train.shape)
print('Test data shape:', x_test.shape)
print('Test labels shape:', y_test.shape)

# Explore class distribution (if using a standard dataset)
print('Class Distribution (Top 10):')
print(Counter(np.argmax(y_train, axis=1)).most_common(10))
```

## Image Preprocessing
#### First, let's normalize the pixel values of the images.

```python 
# Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
```

#### Next, resize images to a consistent size for model input.

````python
# Define batch size
batch_size = 32

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
test_dataset = tf.data.Dataset.from_tensor_slices(x_test)

# Resize images in batches during training
train_dataset = train_dataset.batch(batch_size).map(lambda x: tf.image.resize(x, (128, 128)))

# Resize images in batches during testing
test_dataset = test_dataset.batch(batch_size).map(lambda x: tf.image.resize(x, (128, 128)))
````

## Data Augmentation
#### We'll begin with some basic parameters.
- ##### Experiment with Parameters: The code below has some example data augmentation parameters. Try changing the values within these parameters, or even adding new augmentation techniques! Here's a short guide:
- ##### Hint 1: Start with small adjustments to see the effects clearly.
- ##### Hint 2: Consider which augmentations make sense for your dataset. Flipping images of letters might be okay, but rotating them too much could make them unreadable!
- ##### Explore more: Try adding things like shear_range (for shearing transformations) or zoom_range (for random zooming).
- ##### Visualize the Effects: After setting up your ImageDataGenerator, add a few lines of code to display some randomly augmented images from your dataset. This will help you see how your chosen parameters change the images.
- ##### Hint: Use a small sample of images so it's easy to compare the originals with the augmented versions.

```python
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)
```

#### Now, let's build the model.

````python
# Choose a pre-trained model suitable for object recognition (VGG16, ResNet50, MobileNetV2 are all options)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze some layers of the pre-trained model
for layer in base_model.layers[:-10]:
    layer.trainable = False

num_classes = 10

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
````

#### Compile.

````python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
````

#### Mark a checkpoint.

````python
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
````

#### History, Model Fit

````python
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=20,  # Adjust as needed
    validation_data=(x_test, y_test),
    callbacks=[checkpoint]
)
````

## Model Training
#### Enhanced Training
- ##### Implement data augmentation within the training loop. Add callbacks to monitor progress and save the best performing model. Modify the Training Code: If you haven't already, we need to make a few changes to your training loop:
- ##### Integrate the Data Augmentation: Replace the direct use of x_train with datagen.flow(x_train, y_train, batch_size=32). This will apply your augmentations in real-time during training
- ##### Use the Validation Set: We already have validation_data=(x_test, y_test).
- ##### Save the Best Model: We're using a ModelCheckpoint callback to automatically save the model if its performance on the validation set improves
- ##### Hint: Experiment with different batch sizes as well.

````python
# One-hot encode the target labels
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Define the ImageDataGenerator for data augmentation
datagen_train = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2
)
datagen_train.fit(x_train)

# Define ModelCheckpoint callback to save the best model based on validation accuracy
checkpoint = ModelCheckpoint('best_accuracy_model.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

# Train the model with data augmentation
history = model.fit(
    datagen_train.flow(x_train, y_train, batch_size=32),
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint]
)
````

#### Visualizing Training Progress
- ##### Importance of Monitoring: Explain why tracking validation metrics helps identify overfitting or underfitting.
- ##### Plot training and validation accuracy/loss curves.

````python
# Plot training and validation curves
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.show()

# Plot the loss curves
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()
````

#### Evaluation on the Test Set
- ##### Discuss how test set metrics provide the most unbiased assessment of model performance.

````python
best_model = load_model('best_accuracy_model.h5')
test_loss, test_acc = best_model.evaluate(x_test, y_test)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)
````

#### Hyperparameter Tuning
- ##### Exploring Learning Rates: In the provided code, we're iterating through different learning rates.
- ##### Hint 1: A good starting range for the learning rate is often between 0.01 and 0.0001.
- ##### Hint 2: Pay close attention to how quickly the validation loss starts to increase (if it does), which might signal a learning rate that's too high.

````python
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
````
````python
# Convert the target labels to one-hot encoded format
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

print("Shape of y_train_encoded:", y_train_encoded.shape)
print("Shape of y_test_encoded:", y_test_encoded.shape)
````
````python
def create_model(learning_rate=0.01):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Training with different learning rates
for lr in [0.01, 0.001, 0.0001]:
    model = create_model(learning_rate=lr)
    history = model.fit(x_train, y_train_encoded, epochs=10, validation_data=(x_test, y_test_encoded))
````
````python
# Define evaluate_model function
def evaluate_model(model, x_test, y_test_encoded):

# Assuming y_test is the integer label array
    y_test_encoded = to_categorical(y_test)

# Now, you can pass y_test_encoded to the evaluate_model function
    evaluate_model(model, x_test, y_test_encoded)
````
````python
# Assuming you've trained the model and made predictions on the test data
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print("Shape of y_pred:", y_pred.shape)

print("Shape of y_test_encoded:", y_test_encoded.shape)
````
````python
# Plot the training curves and evaluate the model once more.
def plot_curves(history, learning_rate):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy (LR={learning_rate})')  # Include learning rate in the title
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss (LR={learning_rate})')  # Include learning rate in the title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Evaluate the model
def evaluate_model(model, x_test, y_test):
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)

# Assuming `history` contains the training history of a model and `lr` is the learning rate used
plot_curves(history, lr)

# Assuming `model` is the trained model and `x_test`, `y_test` are the test data
evaluate_model(model, x_test, y_test_encoded)
````

## Confusion Matrix
````python
# Assuming y_test contains the true labels and y_pred_classes contains the predicted labels
# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Print confusion matrix
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Calculate classification report
class_report = classification_report(y_test, y_pred_classes)
print("Classification Report:")
print(class_report)
````

## Finally, evaluate the model on testing data.

````python
# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test_encoded, verbose=0)

# Print the test loss and accuracy
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)
````

## Conclusion

This project demonstrates the complete workflow of image classification using TensorFlow and Keras on the CIFAR-10 dataset. 
By following this example, users can understand how to build, train, evaluate, and visualize CNN models for image classification tasks. 
Future work may involve experimenting with different architectures, optimization techniques, and datasets for further improvement.
