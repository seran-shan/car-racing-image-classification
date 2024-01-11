"""
This module contains the model definitions for the image classification task.
"""
from tensorflow.keras import models, layers, regularizers, callbacks, optimizers


def build_basic_model(input_shape=(96, 96, 3), num_classes=5, learning_rate=0.001):
    """
    Builds a basic CNN model.

    Parameters
    ----------

    input_shape : tuple
        The input shape of the model.
    num_classes : int
        The number of classes to predict.

    Returns
    -------
    model : tensorflow.keras.models.Sequential
        The model.
    """
    print("Creating basic model..")
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(
                128, activation="relu", kernel_regularizer=regularizers.l2(0.01)
            ),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=optimizers.legacy.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# def build_advanced_model(input_shape=(96, 96, 3), num_classes=5, learning_rate=0.001):
#     print("Creating enhanced basic model...")
#     model = models.Sequential(
#         [
#             layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
#             layers.MaxPooling2D(pool_size=(2, 2)),
#             layers.Conv2D(64, (3, 3), activation="relu"),
#             layers.MaxPooling2D(pool_size=(2, 2)),
#             layers.Flatten(),
#             layers.Dense(128, activation="relu"),
#             layers.Dropout(0.3),  # Reduced dropout rate
#             layers.Dense(num_classes, activation="softmax"),
#         ]
#     )

#     # Learning rate scheduler
#     lr_schedule = optimizers.schedules.ExponentialDecay(
#         initial_learning_rate=learning_rate, decay_steps=10000, decay_rate=0.9
#     )
#     opt = optimizers.legacy.Adam(learning_rate=lr_schedule)

#     early_stopping = callbacks.EarlyStopping(
#         monitor="val_loss", patience=10, restore_best_weights=True
#     )

#     model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

#     return model, early_stopping


def build_advanced_model(input_shape=(96, 96, 3), num_classes=5, learning_rate=0.001):
    """
    Builds a more advanced CNN model.

    Parameters
    ----------
    input_shape : tuple
        The input shape of the model.
    num_classes : int
        The number of classes to predict.
    learning_rate : float
        The learning rate of the optimizer.

    Returns
    -------
    model : tensorflow.keras.models.Sequential
        The model.
    """
    print("Creating advanced model..")
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(
                256, activation="relu", kernel_regularizer=regularizers.l2(0.01)
            ),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=optimizers.legacy.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=3)
    return model, early_stopping
