## Pacman Trainer

### The final display with terminal debug output

<img src="https://user-images.githubusercontent.com/1320083/210688412-798b95b4-a958-474a-bbd9-d46a2af7abbf.png"  width="50%" height="50%">

## Prerequisites

- Python 3 (so far, it works with Python 3, but if there are problems, try Python 2)

## Environments Tested

### Windows 10 WSL2 Ubuntu (Build 19042.1288)

- Git Bash
- Python 3.8.10 (Window WSL2 Ubuntu)
- Python 3.9.8 ([See latest versions](https://www.python.org/downloads/))
  - Python must be 3.9.x or earlier or [PyTorch will not work](https://pytorch.org/get-started/locally/#windows-python)

```
sudo apt update
sudo apt-get install python3-venv python3-tk
sudo apt-get install python3-tk
```

### MacOS 11.6 (Build 20G165)

- Python 3.9.4

```
brew install python-tk
```

## Quickstart

```
$ clone this repository
$ cd pacman-agent
$ pip install -r requirements.txt
$ python3 -u environment.py     # Run the app and show the GUI
```

```
$ deactivate  # exit virtual environment
```
## Components

- **PacmanAgent**: Contains all of the logic for the agent controlling Pacman
- **PacmanTrainer**: Contains the logic for training Pacman using Pytorch for deep imitation learning.
- **Environment**: Contains the game logic and is largely structured in the same way as the Wumpus World / Blindbot's environment
  - A constructor, parameterized by the Pacman Trainer's array-of-strings maze representation, that initializes all of the maze variables, as follows:
    - The maze structure, where walls are located, and thus what legal moves are available in any position.
    - The sets and positions of all game actors (beginning with their provided starting positions), including:
      - Pacman
      - Ghosts
      - Pellets
  - Pacman's score (# of pellets eaten)
  - Determining game-ending conditions (viz., Pacman getting eaten by a ghost or eating every pellet)
  - Rendering the maze elements (in their basic array-of-strings format like in the Pacman Trainer) in terms of their graphical equivalents
  - Actors in the environment take turns acting, so the environment determines what happens on every "tick" or turn of a running game:
    - The PacmanAgent's chooseAction method is called with the current game state, and its action choice is enacted (if it's not a legal action, i.e., one that runs it into a wall, it does nothing that turn).
      - If Pacman eats a pellet, it is removed from the board and the score is incremented
      - If Pacman moves onto a ghost's tile, it dies and the game's over
    - All ghosts make an action choice governed by a coin-flip: 10% of the time, it will choose randomly, and the other 90%, it will take a step that brings it closer to Pacman. You can use / adapt the Pathfinder class I gave in the BlindBot package to help with pathfinding or just do something basic like looking at the Manhattan distance between a ghost and Pacman and then choosing the action that minimizes it.
- **MazeUI**: Given a Tk "root window", draw a Tk Canvas on it and all the blocks, characters, and items that make up a maze

## Exercise

To view the accompanying logic binding Pacman Trainer and Agent, see the EXERCISE.md instructions.
