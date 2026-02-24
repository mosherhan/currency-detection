import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from data_manager import get_data_generators
from model import build_model
import config

def train():
    # Ensure models directory exists
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)

    # Get data
    train_gen, val_gen = get_data_generators()
    
    # Build model
    model = build_model(len(config.CLASSES))
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_with_crossentropy' if len(config.CLASSES) > 2 else 'binary_crossentropy', # simplified choice
        metrics=['accuracy']
    )
    # Using categorical_crossentropy for multi-class
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    checkpoint = ModelCheckpoint(
        config.MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        callbacks=[checkpoint, early_stop]
    )

    # Save final model
    model.save(config.MODEL_PATH.replace(".h5", "_final.h5"))
    
    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Loss')
    plt.legend()
    
    plt.savefig('training_results.png')
    plt.show()

if __name__ == "__main__":
    train()
