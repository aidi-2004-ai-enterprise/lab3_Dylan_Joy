# lab3_Dylan_Joy

This project implements a machine learning pipeline using the Seaborn penguins dataset. It trains an XGBoost model, deploys it through a FastAPI application, and The API provides a `/predict` endpoint with input validation/logging.

## Setup

1. Create and activate the virtual environment:

```bash
uv venv
.\.venv\Scripts\activate.bat   # (Windows)
source .venv/bin/activate      # (Linux/macOS)

uv install

---

### 4. **How to Run**

Explain how to run training and launch the API:

```markdown
## Usage

- Train the model:

```bash
python train.py

python -m uvicorn app.main:app --reload

---

### 5. **Testing the API**

Explain briefly how to test the `/predict` endpoint and mention graceful failure:

```markdown
## API Testing

Use the interactive docs to send POST requests to `/predict` with valid penguin features.

Invalid inputs (e.g., wrong sex or island values) will return clear validation errors.

## Demo Video

See the file `demo_requests.mp4` for a screen recording showing successful and failed API requests.
