## FaceSketcher - Words become Faces
<img width="1706" height="877" alt="image" src="https://github.com/user-attachments/assets/f50dc9c4-f481-4f0c-aec7-a59da401d9d3" />


This project currently runs with three backend services and one frontend:

- `module1` (FastAPI, BERT) on port `8000`
- `api` (Flask, LAGAN sketch generator) on port `5001`
- `generator_final` (FastAPI sketch->real image) on port `5002`
- `face-sketcher` (Next.js frontend) on port `3000`

You are using two Python environments:

- `bertenv` (venv) for `module1` and `api`
- `tfenv` (conda) for `generator_final/server.py`

---

## 1) Start all services (Windows)

Open 4 terminals from project root (`Module1`) and run the following.

### Terminal A - BERT service (`module1`)

```powershell
bertenv\Scripts\activate
cd module1
uvicorn app:app --reload --port 8000
```

### Terminal B - LAGAN API (`api`)

```powershell
bertenv\Scripts\activate
cd api
python app.py
```

### Terminal C - Sketch to real image (`generator_final`)

```powershell
conda activate tfenv
cd generator_final
python server.py
```

### Terminal D - Frontend (`face-sketcher`)

```powershell
cd face-sketcher
npm install
npm run dev
```

Open: `http://localhost:3000`

---

## 2) API ports summary

- `http://127.0.0.1:8000` -> `module1` (`/predict`)
- `http://127.0.0.1:5001` -> `api` (`/generate_sketch`)
- `http://127.0.0.1:5002` -> `generator_final` (`/convert`)

If needed, set these in frontend env:

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000/predict
NEXT_PUBLIC_LAGAN_API_URL=http://127.0.0.1:5001
NEXT_PUBLIC_SKETCH2REAL_API_URL=http://127.0.0.1:5002
```

---

## 3) How to share with another person

Do not share your full local virtual environment folders directly (`bertenv`, conda env directory).  
Share the project code plus dependency manifests.

### A. What to send

- Entire project folder (without heavy caches/env folders)
- `requirements.txt` and any service-specific requirements files
- Conda environment export file for `tfenv`
- This README

### B. Export your environments

From your machine:

```powershell
# from project root
pip freeze > requirements-bertenv.txt
conda env export -n tfenv > tfenv.environment.yml
```

Share these two files with the project.

### C. Recreate environments on another machine

#### Recreate `bertenv`

```powershell
python -m venv bertenv
bertenv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-bertenv.txt
```

#### Recreate `tfenv`

```powershell
conda env create -f tfenv.environment.yml
conda activate tfenv
```

Then run the same startup commands from section 1.

---

## 4) Recommended folders to exclude while sharing

- `bertenv/`
- `.venv*/`
- `__pycache__/`
- `.next/`
- `node_modules/`
- large model output folders unless required

---

## 5) Notes

- Keep model files/checkpoints in expected paths for each service.
- If a service fails on startup, check Python environment first (`where python`, `python --version`).
- `generator_final` should run only from `tfenv` (TensorFlow environment).
