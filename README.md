# ManPac: Reinforcement Learning for Pac-Man
This readme contains information regarding the project structure and runtime instructions.

## Project structure
The project structure is as follows:
```
pacman-rl/

├── models/
├── notebooks/
├── src/
├── environment.yml
├── README.md
```
The project structure includes several key folders and files:
- `notebooks`: Contains Jupyter notebooks
- `src`: Holds the source code
- `models`: Stores models' information
- `environment.yml`: Creates a conda environment with necessary packages
- `README.md`: Provides project information


## Installation
To install the necessary packages, use the provided conda environment file. First, make sure you have conda installed. If you don't have it installed, you can download it from [here](https://docs.conda.io/en/latest/miniconda.html).

To create a new conda environment from the environment file, use the following command:
```bash
conda env create -f environment.yml
```

To activate the environment, use:
```bash
conda activate PacmanOpenAIGym_FHDortmund
```
If you're using an IDE like PyCharm or VSCode, you have to select the environment in the IDE settings.

## Running Jupyter Lab
Instead of using Jupyter notebooks, we recommend using Jupyter Lab. To run Jupyter Lab, use the following command:
```bash
jupyter lab
```
The cell should then output a link that you can open in your browser to access Jupyter Lab. Check the output in the terminal for the link. It should look something like this:
```
http://localhost:8888/lab?token=aa7e97c2d09bae01632db508730ab48cbc4609fc92c7fc6e
```
The token can vary, so make sure to use the one provided in the output.

## Models
The models folder contains all the useful information for each different model and level of the game.

The `training_plots` folder contains plots of different hyperparameter combinations and their respective rewards.

Using the training plots, for each level fine-tuned hyperparmeters have been saved in the `hyperparameters.ini` file which you can easily copy and replace with the one in the `src` folder to train and play the model.

Finally, each level folder also has a saved model (with the .pkl extension) that you can easily load up and have it play any level. 

## Usage
First of all, activate the environment:


```bash
conda activate PacmanOpenAIGym_FHDortmund
```

To train a model and play the game, first set the desired level and model hyperparameters in `src/gym-pacman-environment/hyperparameters.ini` (for Dyna-Q you can omit the hyperparameters after `discount_factor`). Then run the `src/gym-pacman-environment/train_and_play.py` file.

```bash
python src/gym-pacman-environment/train_and_play.py
```

To play from a saved model, open `src/gym-pacman-environment/play_saved_model.py`, adjust the `filename` variable, and then run the file.


&copy; 2025 Hazhir Amiri, Raed Kayali, Lucas Schönhold, 