# Contributing to ARGUS Stream A

Welcome! This document outlines guidelines and instructions for developers contributing to the **ARGUS Stream A** project. 

---

## 1. Project Organization

Our project structure follows a clean separation of concerns, dividing core research logic from hosted application endpoints:

```text
configs/                 Dataset recipes and training configurations
deployment/              ASGI APIs, Modal GPU deployment classes, and Vercel Next.js app
docs/                    Methodology, evaluations, audits, and briefs
src/                     Machine learning library (data, models, evaluation, utilities)
  └── inference/         Core InferenceEngine, InferenceProfile, and shared routines
tests/                   API contract and deployment verification test suites
demo.py                  Local Gradio UI for quick interactive visualization
```

*Note: All core model scoring and video preprocessing logic belongs in `src/`. The scripts in `deployment/` or `demo.py` are strictly for presentation.*

---

## 2. Local Setup

### Prerequisites
- Python 3.10 or 3.11
- Node.js 18 or 20 (for frontend)
- Git LFS (if checking out heavy checkpoint assets)

### Backend Environment Setup
1. Clone the repository and navigate to the project directory.
2. Initialize and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Local Demos
- To run the interactive Gradio demo locally:
  ```bash
  python demo.py
  ```
- To run the FastAPI server locally:
  ```bash
  python deployment/app.py
  ```

---

## 3. Deployment Configuration

### Modal GPU Backend
Our backend runs serverless on Modal with scale-to-zero settings for cost management.

1. Install and authenticate the Modal CLI:
   ```bash
   pip install modal
   modal setup
   ```
2. Prime the Hugging Face cache (downloads VideoMAE backbone weights to the persistent volume):
   ```bash
   modal run deployment/modal_app.py --prime-cache
   ```
3. Deploy the API:
   ```bash
   modal deploy deployment/modal_app.py
   ```

### Vercel Next.js Frontend
1. Navigate to the vercel workspace:
   ```bash
   cd deployment/vercel_app
   ```
2. Install npm packages:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
4. Build verification:
   ```bash
   npm run build
   ```

---

## 4. Testing & Code Quality

### Running Tests
We use `pytest` for Python testing and Next.js compilation for frontend checks.
```bash
# Run python tests
pytest tests/ -v

# Run frontend build check
cd deployment/vercel_app && npm run build
```

### Formatting Standards
- **Python**: We use `ruff` for linting and code formatting. Before committing, format and check your code:
  ```bash
  ruff check .
  ruff format .
  ```
- **TypeScript/React**: Ensure all TypeScript compile-time errors are resolved prior to staging files.
