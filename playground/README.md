# CDM-1 Developer Playground

The **CDM-1 Developer Playground** is a modern, web-based visual interface designed for developers, researchers, and operators to interact with, evaluate, and inspect the CDM-1 Cognitive Decision Model and Autonomous Agent Loop without writing Python code.

---

## Features

- **System Dashboard**: Live node readiness, PyTorch device & hardware detection, 158M parameter checkpoint SHA-256 verification, and SQLite persistence enclave status.
- **Cognitive Playground**: Interactive task execution with context narrative, structured JSON features, candidate actions, and safety constraints. Visualizes real-time pipeline traces (Perception → Memory → Reasoning → Plan Synthesis → Calibration).
- **Decision Explorer**: Test multi-task decision gates (binary classification, candidate ranking, continuous scoring, outcome risk prediction, and calibrated abstention).
- **Agent Runs & Approval Gate**: Monitor autonomous agent control loops with step budgets, cost caps, execution logs, and human-in-the-loop one-click approval/rejection gates.
- **Memory Explorer**: Live inspection of SQLite episodic memory, semantic domain rules, procedural patterns, and world-state telemetry without fabrication.
- **Diagnostics**: Real-time system health, PyTorch hardware checks, model signature verification, and troubleshooting guides.

---

## Quick Start (One Command)

From the project root directory, run:

```bash
python scripts/start_playground.py
```

Or on Windows:
```cmd
start_playground.bat
```

This single command:
1. Validates Python and Node.js prerequisites.
2. Installs frontend dependencies if needed (`npm install`).
3. Launches the FastAPI backend (`http://127.0.0.1:8000`).
4. Launches the Next.js Developer Playground (`http://localhost:3000`).
5. Opens your default browser automatically.

---

## Manual Execution

If you prefer running services independently:

### 1. Start the CDM-1 REST Backend
```bash
python api/main.py
```
Backend will be available at `http://127.0.0.1:8000` (API docs at `/docs`).

### 2. Start the Frontend
```bash
cd playground
npm run dev
```
Playground interface will be accessible at `http://localhost:3000`.

---

## Tech Stack

- **Framework**: Next.js 16 (App Router), React 19, TypeScript
- **Styling**: Tailwind CSS v4
- **Icons**: Lucide React
- **Data Visualization**: Recharts
- **API Client**: Strongly-typed client in `playground/lib/cdm-api.ts` connecting to CDM-1 FastAPI endpoints.
