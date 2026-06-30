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
- **Live Data Integration**: Fetches real-time, in-progress World Cup 2026 scores. If a match is finished, the real score/outcome (including penalty shootouts) overwrites the prediction, and the bracket updates dynamically!
- **Historical Snapshots**: Every run of the engine is saved. You can instantly jump back to past predictions to see how the model's expectations changed as the tournament progressed.
- **Deep Match Analysis**: View head-to-head records (with time decay), Expected Goals (xG), and exact scoreline probabilities.
- **Outdated Data Warning**: The Web UI automatically alerts you with a warning banner if a scheduled match kickoff time has passed (+3 hours match time) and the local data hasn't been updated yet.
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
│   ├── index.html           # Dashboard / Homepage (includes outdated check)
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
- Fetch over 49,000 matches, compute the math, and save the data.

**Step 2:** Once completed, press `2` and hit Enter. Your default browser will instantly open the beautiful AMPI dashboard!

---

## The Statistical Model (Deep Dive)

AMPA utilizes industry-standard statistical modelling used by professional sports analysts.

### 1. Custom Elo Rating System
At the heart of the engine is a custom Elo algorithm. Starting from a base rating, every time two teams play, points are exchanged.
- **Goal Difference Multiplier**: Winning 4-0 yields significantly more points than winning 1-0.
- **Tournament Weighting**: Friendlies have a very low weight. World Cup matches have the highest weight. Beating France in a World Cup Final yields massive Elo gains compared to beating them in a friendly.

### 2. Opponent-Quality Adjusted Attack & Defense Indexes
Rather than relying on raw weighted goal averages, AMPA computes an **Attack Index** and **Defense Index** that are dynamically adjusted for opponent quality:
- **Opponent Strength Weighting**: Scoring a goal against a defensive powerhouse like France is weighted significantly higher than scoring against a defensive liability.
- **Time Decay**: A goal scored in 2026 is worth 100% weight. A goal scored in 2022 is worth 50% weight. This ensures the model reacts to current squad generations, not past glory.

### 3. Negative Binomial Distribution (The Simulator)
To resolve the classical overdispersion problem in football goal modeling (where the variance of goals is greater than the mean, resulting in more draws and heavy scorelines than standard models predict), AMPA uses a **Negative Binomial Distribution** (replacing the basic Poisson model):
- **Dixon-Coles Correction**: A low-score correction factor is applied to discourage 0-0 inflation and balance 1-0/0-1 predictions.
- **Bivariate Urgency**: When both teams score, a custom correlation factor is applied to reflect open, aggressive play styles, resulting in highly realistic match outcomes.
- **Clean Outcome Types**: Predicted scorelines reflect the 90-minute score, while extra-time results (e.g. `(Belgium wins in ET)`) or penalty shootouts (e.g. `(Morocco wins on Pens)`) are calculated and appended clearly as text.

### 4. Form, Streaks & Momentum
- **Form Score**: The engine looks at a team's last 10 matches, weighting the most recent matches exponentially higher.
- **Momentum Engine**: Scan the last 15 matches to trigger momentum multipliers: win streaks (4+ games), unbeaten streaks (8+ games), scoring streaks (consecutive 2+ goal games), and losing streaks (3+ losses).
- **Recent H2H Nudge**: If two teams have faced each other within the last 180 days (e.g., in the Group Stage), the winner of that recent matchup receives a substantial confidence boost, which decays over a 60-day half-life.

### 5. Tournament Pressure, Fatigue & Venue Factors
- **Knockout Stage Pressure**: Teams with a strong historical track record in major tournament knockout-stage matches (e.g., Argentina, France) receive a slight edge.
- **Fatigue & Rest Days**: During tournament simulation, the engine calculates the rest days since each team's last match. Teams with fewer rest days receive a performance penalty.
- **Venue & Travel Factor**: Hosts (USA, Canada, Mexico) receive a major home advantage boost. Other teams receive travel modifiers based on their continental confederation (e.g. CONCACAF/CONMEBOL teams receive a slight familiarity boost, while AFC/CAF teams receive a minor travel penalty).
- **FIFA Ranking Anchor**: The engine fetches official FIFA World Rankings as a soft anchor to ensure the Composite Rating remains grounded in reality.

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
