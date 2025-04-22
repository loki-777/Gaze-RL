# Gaze-RL

Implementation for "Deep Decision Making with RL" course project titled "Gaze-Guided RL for Object Search in AI2-THOR"

## Table of Contents

-   [Project Structure](#project-structure)
-   [Setup and Installation](#setup-and-installation)
-   [Usage](#usage)

## Project Structure

```bash
Gaze-RL/
├── configs/
├── notebooks/
│   └── ai2thor_exploration.ipynb
├── src/
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── ai2thor_env.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── agents.py
│   │   ├── gaze_predictor.py
│   │   └── networks.py
│   │── __init__.py
│   │── train.py
│   │── eval.py
│   └── utils.py
├── requirements.txt
├── .gitignore
└── README.md

```

## Setup and Installation

```bash
pip install -r requirements.txt
```

## Usage

<!-- ```bash
cd Gaze-RL/
python -m src.train configs/<config_name>.yaml
``` -->
