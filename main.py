"""
This module contains the model definitions for the image classification task.
"""
import argparse
import numpy as np
import gymnasium as gym
from tensorflow.keras.models import load_model
from data_preprocessing import load_and_preprocess_images
from cnn import build_advanced_model, build_basic_model

config = {
    "epochs": 20,
    "batch_size": 32,
    "seed": 2000,
    "input_shape": (96, 96, 3),
    "num_classes": 5,
    "learning_rate": 0.001,
    "eval_model": "model_20.keras",
}


def evaluation(env: gym.Env, model):
    """
    Evaluate the model using the enviroment.

    Parameters
    ----------
    env : gym.Env
        The environment to play the game in.
    model : tensorflow.keras.models.Sequential
        The model to use for playing.
    """
    # Initialize and skip initial frames
    obs, _ = env.reset(seed=config["seed"])
    for _ in range(50):
        obs, _, _, _, _ = env.step(0)

    # Play loop
    done = False
    while not done:
        obs_batch = np.expand_dims(obs, axis=0)
        action = np.argmax(model.predict(obs_batch))
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated


def train(model, early_stopping=None):
    """
    Train the model.

    Parameters
    ----------
    model : tensorflow.keras.models.Sequential
        The model to train.
    early_stopping : tensorflow.keras.callbacks.EarlyStopping
        The early stopping callback.
    """
    print("Starting training process...")
    train_images, train_labels = load_and_preprocess_images("data/raw/train")

    # Split the data into training and validation sets
    val_split = 0.2
    num_val_samples = int(len(train_images) * val_split)
    training_images = train_images[:-num_val_samples]
    training_labels = train_labels[:-num_val_samples]
    validation_images = train_images[-num_val_samples:]
    validation_labels = train_labels[-num_val_samples:]

    if early_stopping:
        model.fit(
            training_images,
            training_labels,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            validation_data=(validation_images, validation_labels),
            callbacks=[early_stopping],
        )
    else:
        model.fit(
            training_images,
            training_labels,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            validation_data=(validation_images, validation_labels),
        )

    model_type = "advanced" if early_stopping else "basic"
    print("Training completed. Saving the model...")
    model_path = f"models/{model_type}/model_{config['epochs']}.keras"
    model.save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument(
        "--model", type=str, choices=["basic", "advanced"], help="Model to use"
    )
    args = parser.parse_args()

    if args.eval:
        if not args.model:
            raise ValueError("Please specify a model to evaluate")
        env_args = {
            "domain_randomize": False,
            "continuous": False,
            "render_mode": "human",
        }
        print("Initializing environment...")
        env = gym.make("CarRacing-v2", **env_args)

        print(f"Environment: {env.spec.id}")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        model = load_model(f"models/{args.model}/{config['eval_model']}")
        evaluation(env, model)
    elif args.train:
        if args.model == "basic":
            model = build_basic_model(
                config["input_shape"], config["num_classes"], config["learning_rate"]
            )
            train(model)
        elif args.model == "advanced":
            model, early_stopping = build_advanced_model(
                config["input_shape"], config["num_classes"], config["learning_rate"]
            )
            train(model, early_stopping)
