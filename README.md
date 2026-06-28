<div align="center">
  <h1>Agnel Match Predicting Algorithm (AMPA)</h1>
  <p><strong>A Next-Generation Statistical Football Predictor for the 2026 FIFA World Cup</strong></p>
  
  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
  [![Status](https://img.shields.io/badge/Status-Active-success.svg)]()
</div>

---

## Table of Contents
1. [Introduction](#introduction)
2. [Core Components](#core-components)
3. [Features](#features)
4. [Project Architecture](#project-architecture)
5. [Installation & Setup](#installation--setup)
6. [The Statistical Model (Deep Dive)](#the-statistical-model-deep-dive)
    - [Custom Elo Rating System](#1-custom-elo-rating-system)
    - [Time-Decaying Attack & Defense Indexes](#2-time-decaying-attack--defense-indexes)
    - [Bivariate Poisson Distribution](#3-bivariate-poisson-distribution-the-simulator)
    - [Form & Tournament Pedigree](#4-form-score--tournament-pedigree)
    - [Dynamic Host Advantage](#5-dynamic-host-advantage--anchor-weights)
7. [Data Sources](#data-sources)
8. [Legal & Disclaimer](#legal--disclaimer)

---

## Introduction

Welcome to **AMPA (Agnel Match Predicting Algorithm)**, a state-of-the-art simulation framework engineered specifically for the 2026 FIFA World Cup. 

Unlike basic predictors that rely purely on static FIFA rankings or simple win/loss ratios, AMPA processes over **49,000 historical international matches dating back to the 1870s**, computes dynamic attacking and defensive strengths, and runs a complex **Bivariate Poisson Distribution** to simulate every possible scoreline of every match in the tournament.

Whether you are a data scientist, a football fanatic, or just want to see who will lift the trophy in 2026, AMPA provides an unparalleled level of statistical depth wrapped in a beautiful interface.

---

## Core Components

This project is divided into two distinct, strictly separated architectures:

1. **AMPE (The Engine)**: A headless Python backend that acts as the brain. It fetches datasets, cleans data, computes ratings, runs the probabilistic simulations, and outputs the results as timestamped JSON payloads.
2. **AMPI (The Interface)**: A vanilla HTML/JS web application (Agnel Match Prediction Interface) that reads the engine's JSON payloads. It visually renders the tournament bracket, displays comprehensive team profiles, and provides a deep dive into individual match predictions.

---

## Features

- **End-to-End Simulation**: Predicts the entire tournament from the Round of 32 to the Final.
- **Live Data Integration**: Fetches real-time, in-progress World Cup 2026 scores. If a match is finished, the real score overwrites the prediction, and the bracket updates dynamically!
- **Historical Snapshots**: Every run of the engine is saved. You can instantly jump back to past predictions to see how the model's expectations changed as the tournament progressed.
- **Deep Match Analysis**: View head-to-head records, Expected Goals (xG), and exact scoreline probabilities (e.g., a 12% chance of a 2-1 victory).
- **Automated Dependency Management**: A completely seamless Windows batch interface (`ampe.bat`) handles virtual environments, dependencies, and execution automatically.

---

## Project Architecture

AMPA follows a strict separation of concerns, ensuring backend logic, frontend UI, and generated data remain completely modular.

```text
fifa-predictor/
├── ampe.bat                 # Master Launcher Menu (Double-click to start!)
├── engine/                  # Backend Simulation Core
│   ├── main.py              # The core AMPE simulation script
│   └── ampe_helper.py       # Helper script for managing prediction saves
├── ui/                      # Frontend HTML Interface
│   ├── index.html           # Dashboard / Homepage
│   ├── bracket.html         # Visual Tournament Tree
│   ├── match.html           # Deep dive into a specific matchup
│   ├── runs.html            # Archive of past predictions
│   ├── team.html            # Individual team statistics
│   ├── teams.html           # Global ranking leaderboard
│   └── setup.html           # Fallback setup instructions
└── data/                    # The Database (Generated automatically)
    ├── data.js              # The active/latest prediction payload
    ├── runs_index.js        # The registry of all generated runs
    └── outputs/             # Archived snapshots (e.g., data_20260628_103505.js)
```

---

## Installation & Setup

AMPA is designed to be incredibly easy to run, completely eliminating the need for manual command-line execution.

### Prerequisites
- **Windows OS** (Due to the automated batch script)
- **Python 3.10+** (Ensure Python is added to your PATH during installation)

### Running the Engine

1. Download or clone this repository.
2. Navigate into the `fifa-predictor` folder.
3. Double-click the **`ampe.bat`** file.

You will be greeted with the master menu:

```text
==========================================================
       AGNEL MATCH PREDICTING ALGORITHM (AMPA)
==========================================================

1. Run Prediction Engine (AMPE)
2. Launch Interface (AMPI)
3. View Saves
4. Delete a Save
5. Exit
```

**Step 1:** Press `1` and hit Enter. The script will automatically:
- Create an isolated Python virtual environment (`venv/`).
- Install all heavy scientific dependencies (`pandas`, `numpy`, `scipy`).
- Fetch over 40,000 matches, compute the math, and save the data.

**Step 2:** Once completed, press `2` and hit Enter. Your default browser will instantly open the beautiful AMPI dashboard!

---

## The Statistical Model (Deep Dive)

AMPA isn't just a random number generator. It utilizes industry-standard statistical modelling used by professional sports analysts.

### 1. Custom Elo Rating System
At the heart of the engine is a custom Elo algorithm. Starting from a base rating, every time two teams play, points are exchanged.
- **Goal Difference Multiplier**: Winning 4-0 yields significantly more points than winning 1-0.
- **Tournament Weighting**: Friendlies have a very low weight. World Cup matches have the highest weight. Beating France in a World Cup Final yields massive Elo gains compared to beating them in a friendly.

### 2. Time-Decaying Attack & Defense Indexes
Elo is great for overall ranking, but it doesn't tell us *how* a team plays. AMPA computes an **Attack Index** and **Defense Index** for every team relative to the global average.
- **Time Decay**: A goal scored in 2026 is worth 100% weight. A goal scored in 2022 is worth 50% weight. A goal scored in 2012 is worth almost nothing. This ensures the model reacts to current squad generations, not past glory.

### 3. Bivariate Poisson Distribution (The Simulator)
When Team A plays Team B, AMPA calculates:
- `Team A xG` = (Team A Attack Index) × (Team B Defense Index) × (Global Average Goals)
- `Team B xG` = (Team B Attack Index) × (Team A Defense Index) × (Global Average Goals)

However, football goals are not entirely independent (if Team A scores 3, Team B is likely pushing higher up the pitch, increasing their chance of scoring). AMPA uses a **Bivariate Urgency** correlation factor. This slightly increases the probability of draws (1-1, 2-2) and reduces the likelihood of stale, independent 1-0 scorelines, resulting in highly realistic match outcomes.

### 4. Form Score & Tournament Pedigree
- **Form**: The engine looks at a team's last 5-10 matches. A high win-rate yields a percentage boost (momentum).
- **Pedigree**: Teams with a historical track record of winning major tournament knockout games (e.g., Argentina, France, Germany) receive a slight "clutch" factor boost, representing their experience under pressure.

### 5. Dynamic Host Advantage & Anchor Weights
- **Home Turf**: The 2026 hosts (USA, Canada, Mexico) receive a mathematically calculated 15% Elo boost. Other CONCACAF teams receive a minor 5% regional familiarity boost.
- **Anchoring**: To prevent statistical anomalies (e.g., a small nation farming Elo points against other small nations), the engine fetches the live **Official FIFA World Rankings**. This acts as a soft anchor, ensuring the model's Composite Rating stays grounded in reality.

---

## Data Sources

AMPA stands on the shoulders of incredible open-source datasets:

1. **Historical Matches**: [martj42/international_results](https://github.com/martj42/international_results) - The most comprehensive dataset of international football matches.
2. **Official FIFA Rankings**: [tadhgfitzgerald/fifa_ranking](https://github.com/tadhgfitzgerald/fifa_ranking) - Used as the anchoring metric for composite ratings.
3. **Live 2026 Scores**: `worldcup26.ir` - A live REST API providing real-time data for the 2026 tournament, ensuring the engine reacts to real events as they happen.

---

## ⚖️ Legal & Disclaimer

**AMPA IS NOT FINANCIAL ADVICE, NOR IS IT A SPORTS BETTING TOOL.** 
The creator(s) and maintainers assume **ZERO RESPONSIBILITY or LIABILITY** for any financial losses or damages resulting from the use of this software. Football is unpredictable, and this engine provides probabilistic estimations, not guarantees.

For full details regarding third-party API usage and our "As-Is" software policy, please read the full [Legal Disclaimer](docs/legal.md).

---

<div align="center">
  <i>Developed with passion for the love of the Beautiful Game.</i>
</div>
