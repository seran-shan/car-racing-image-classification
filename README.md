# CarRacing-v2 Image Classification Project

Welcome to the CarRacing-v2 Image Classification Project! This project is designed to solve an image classification problem to learn the behavior of a racing car in a Gym environment. Our focus is on developing and training CNN models to classify 96x96 color images of the racing environment, labelling them with one of the 5 available actions for car control.

## Project Structure

### Files

- **main.py**: The core script of the project. Use this to train the model and play the game with the trained model.
- **cnn.py**: This file contains the functions to build different CNN models. It includes two distinct approaches for the architecture.
- **data_preprocessing.py**: Contains the function to load and preprocess images, ensuring they are ready for model training.
- **play_policy_template.py**: Provides a template for the play policy to be used in the game environment.
- **utils.py**: A collection of utility functions that support various operations within the project.

### Models

- Stored in the `models/` directory. After training, models are saved here for later evaluation or further use.

### Data

- Located in the `data/` directory.
  - `raw/`: Contains the raw images as obtained from the environment.
  - `processed/`: Houses the preprocessed images ready for model training.

## Usage

To use the `main.py` script, you can run it with different arguments to perform specific tasks:

1. **Training the Model**

   ```
   python main.py --train --model [basic or advanced]
   ```

   Use `--model basic` for the first CNN architecture or `--model advanced` for the second.

2. **Evaluating the Model**
   ```
   python main.py --eval --model [basic or advanced]
   ```
   This will load the trained model and evaluate its performance in the Gym environment.

## Get Started

To get started with this project, clone the repository and install the required dependencies. Ensure you have the necessary Python packages like `tensorflow`, `gymnasium`, and `opencv-python` installed.

After setting up, you can train your model using `main.py` and modify the CNN architectures in `cnn.py` as per your requirements.
