# Developer Setup & Contribution Guide

This document is an exhaustive, deep-level guide for developers, data scientists, or power-users who wish to completely bypass the `ampe.bat` wrapper, manually execute the engine, manipulate the mathematical parameters, debug the API endpoints, or extend the project's capabilities.

---

## 1. Manual Environment Setup

The `ampe.bat` script abstracts away the Python virtual environment. If you want to develop the engine, you must interface with it manually.

### 1.1 Creating the Isolated Environment
It is strictly advised not to install the dependencies globally, as `pandas` and `scipy` can clash with system-level packages.

```bash
# Navigate to the root directory
cd fifa-predictor

# Create the virtual environment
# Windows
python -m venv venv
# Linux / macOS
python3 -m venv venv
```

### 1.2 Activating the Environment
You must activate the environment before running any scripts.

```bash
# Windows (Command Prompt)
.\venv\Scripts\activate.bat

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```
*Note: If you receive a PowerShell Execution Policy error on Windows, run: `Set-ExecutionPolicy Unrestricted -Scope CurrentUser`.*

### 1.3 Installing Dependencies
```bash
pip install pandas numpy scipy requests
```

---

## 2. Advanced CLI Execution

The engine (`engine/main.py`) supports command-line arguments that are normally hidden by the batch script.

### 2.1 Standard Execution (Verbose Mode)
```bash
python engine/main.py
```
Running this without flags will trigger **Verbose Mode**. The terminal will output an immense amount of debug data:
- It will print the detailed mathematical profile of all 32 World Cup teams (Elo, Attack Index, Defense Index, Form %, Composite Score).
- It will print out every single matchup, round-by-round, displaying the exact Expected Goals (xG) calculations and the probabilistic outcome of the Bivariate Poisson distribution.
- Use this mode when tweaking the math to see exactly how your changes impact individual matches.

### 2.2 Quiet Mode (Production Mode)
```bash
python engine/main.py --quiet
```
This is the command executed by `ampe.bat`. It suppresses the massive wall of text, only outputting top-level loading indicators (e.g., "Downloading CSV", "Computing Elo") and the final Predicted Champion.

---

## 3. Manipulating the Statistical Parameters

If you wish to fork AMPA and alter the underlying statistics, all core mathematical weights are located inside `engine/main.py`.

### 3.1 Adjusting the Elo `K` Factor
Search for the `compute_elo()` function. You will find a dictionary or conditional block defining the `K` weight.
- To make friendlies matter even less, reduce the friendly weight from `20` to `5`.
- To make the World Cup highly volatile, increase the World Cup weight from `60` to `100`.

### 3.2 Altering the Time-Decay and Model Constants
Search for the constants block at the top of `engine/main.py`.
- **`HALF_LIFE_DAYS`**: Controls the goal decay rate (default: `180` days). Lowering this makes the engine extremely short-sighted (caring only about the last few months).
- **`NBINOM_R`**: Controls the Negative Binomial dispersion parameter (default: `5.0`). Lowering this increases the variance ("fatter tails"), predicting more extreme blowouts and draws.
- **`GOAL_INFLATE`**: Scaling factor (default: `1.25`) that adjusts predicted match goal rates up to realistic international levels.

### 3.3 Hacking the Venue & Travel Factor
Search for the `compute_venue_factor()` function.
- It dynamically assigns modifiers based on team confederation (e.g. `1.12` for local hosts, `1.05` for CONCACAF, down to `0.96` for AFC teams representing long-distance travel fatigue).
- You can override these modifiers or set them all to `1.0` to completely simulate the tournament on completely neutral soil.

---

## 4. API Endpoints & Data Injection

### 4.1 Live API Polling
The engine attempts to fetch live scores from `https://worldcup26.ir/get/games`. 
If you are developing without internet access, or if the API goes down, the script contains a `try-except` block that gracefully falls back to pure prediction mode, ignoring live scores.

### 4.2 Extending the Data Schema
If you add a new metric to the Python output (e.g., adding `player_injuries` to the team profile in `main.py`), it will successfully serialize into `data.js`. 
However, **it will not magically appear in the UI**. You must open the corresponding HTML file (e.g., `ui/team.html`), locate the Javascript rendering block, and explicitly add your new variable to the DOM:

```javascript
// Example modification in ui/team.html
document.write('<div class="stat">Injuries: ' + p.player_injuries + '</div>');
```

---

## 5. Troubleshooting Common Developer Errors

- **`ModuleNotFoundError: No module named 'pandas'`**: You forgot to activate your virtual environment before running `main.py`.
- **`PermissionError: [Errno 13] Permission denied: '../data/data.js'`**: The frontend UI (or your IDE) currently has `data.js` locked open. Close your browser tab or IDE file and re-run.
- **UI shows "System Setup Required" despite running the script**: You ran `main.py` from the wrong directory. Ensure your terminal's current working directory is the root `fifa-predictor` folder, so the script correctly resolves `os.path.join(out_dir, '..', 'data')` and places the JSON payload in the right folder.
