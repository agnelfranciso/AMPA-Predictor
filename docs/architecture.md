# AMPA System Architecture: Deep Dive

The Agnel Match Predicting Algorithm (AMPA) was engineered with a strict decoupling of backend simulation logic and frontend rendering. This document provides an exhaustive breakdown of the data flow, folder structure, UI rendering pipeline, component responsibilities, and the architectural philosophies behind the project.

---

## 1. Architectural Philosophy: Why a Static JSON/JS Database?

The most unique architectural decision in AMPA is the complete lack of a traditional SQL or NoSQL database (e.g., PostgreSQL, MongoDB, or even SQLite), and the absence of a localized web server (e.g., Node.js Express, Python Flask).

### The Problem
Traditional data science projects require the user to boot up a backend server to serve REST APIs to the frontend. This creates immense friction for casual users or analysts who just want to double-click an `index.html` file and view the bracket. However, modern web browsers enforce strict **CORS (Cross-Origin Resource Sharing)** policies, meaning a local `index.html` file cannot simply use `fetch('data.json')` from the local hard drive without a web server.

### The AMPA Solution
AMPA circumvents this by utilizing **Javascript Data Injection**. 
When the Python engine (`main.py`) finishes a simulation run, it dumps the massive JSON object into a `.js` file, explicitly assigning it to a global variable:

```javascript
// Inside data/data.js
const PREDICTOR_DATA = { ... massive JSON payload ... };
```

The UI (`index.html`) simply loads this script synchronously:
```html
<script src="../data/data.js"></script>
```
By the time the DOM loads, the `PREDICTOR_DATA` object is globally available in the browser's memory. This entirely bypasses CORS restrictions, requires zero local web servers, and allows the entire UI to operate as a pure, static, double-clickable offline web app.

---

## 2. Directory & Component Breakdown

```text
fifa-predictor/
├── ampe.bat                 # The Master Orchestrator
├── engine/                  # AMPE (The Backend)
│   ├── main.py
│   └── ampe_helper.py
├── ui/                      # AMPI (The Frontend)
│   ├── index.html
│   ├── bracket.html
│   ├── match.html
│   ├── runs.html
│   ├── team.html
│   ├── teams.html
│   └── setup.html
└── data/                    # The "Database"
    ├── data.js
    ├── runs_index.js
    └── outputs/
        ├── data_20260628_103505.js
        └── data_20260628_110522.js
```

### 2.1 The Engine (`engine/`)
The engine is the heavy lifter, written entirely in Python to leverage scientific computing libraries (`pandas`, `numpy`, `scipy`).

- **`main.py`**: This is the monolithic script. It handles everything sequentially:
  1. **Data Ingestion**: Downloads the latest historical CSVs and queries the live `worldcup26.ir` REST API.
  2. **Data Cleansing**: Normalizes team names across 5 different datasets (e.g., ensuring "USA", "United States", and "USA Men" all map to the same entity).
  3. **Computation**: Runs the Elo permutations and time-decay algorithms over the 49,000 matches.
  4. **Simulation**: Iterates through the World Cup bracket, running the Bivariate Poisson distribution for every single fixture.
  5. **Serialization**: Formats the final output and writes it to `../data/data.js`.

- **`ampe_helper.py`**: A lightweight CLI utility script. It does not run predictions. Instead, it reads the `runs_index.js` file, parses it, and provides terminal-friendly outputs for the `ampe.bat` menu (allowing users to view and delete saves without opening the UI).

### 2.2 The Interface (`ui/`)
The interface (AMPI) is incredibly lightweight, built with vanilla HTML, CSS, and JS to ensure maximum performance without the overhead of React, Vue, or Angular.

- **Dynamic Loading via URL Params**: 
  The UI is capable of viewing the past. If a user navigates to `index.html?run=20260628_103505`, the Javascript intercepts the query parameter before DOM load, and dynamically alters the `<script>` tag to point to `../data/outputs/data_20260628_103505.js` instead of the main `data.js`. This allows the UI to render a complete "snapshot" of a historical prediction perfectly.

- **Client-Side Rendering**: 
  All table sorting (e.g., ranking teams by Composite Score in `teams.html`), flag rendering, and bracket line-drawing is done dynamically on the client side using the loaded JSON payload.

### 2.3 The Data Layer (`data/`)
- **`data.js`**: The absolute latest run of the engine. This is what the UI loads by default.
- **`runs_index.js`**: An array of metadata summarizing every single run ever executed. The engine unshifts new runs into this array. The frontend `runs.html` page uses this to display the archive list.
- **`outputs/`**: A permanent historical archive. Every time the engine runs, a unique timestamped copy of `data.js` is permanently stored here.

---

## 3. Data Flow Pipeline

1. **Trigger**: User runs `ampe.bat` and selects `[1]`.
2. **Ingest**: `main.py` fetches the live tournament state from `worldcup26.ir`.
3. **Simulate**: The engine crunches the numbers. If a live match is marked as `finished: true`, the engine halts the Poisson prediction for that specific match and hardcodes the actual real-world score into the bracket.
4. **Cascade**: The real-world score cascades through the mathematical model, physically altering the matchups in the subsequent rounds (Round of 16, Quarter Finals).
5. **Write**: The engine dumps the JSON to `data.js`, archives a copy in `data/outputs/`, and updates `runs_index.js`.
6. **Render**: The user selects `[2]` in `ampe.bat`. The browser opens `ui/index.html`, instantly pulling the fresh `data.js` into memory and rendering the updated reality.
