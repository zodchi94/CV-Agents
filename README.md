# CV Agent System (Resume Optimizer)

A multi-agent LLM-powered system for refactoring and tailoring resumes to specific job vacancies via the OpenRouter API. The system analyzes vacancy requirements, generates multiple resume variants in parallel, validates them for factual accuracy (audit), evaluates relevance (scoring), and iteratively improves the result based on meta-criticism (critic).

---

## 🛠️ System Requirements

To deploy and run the application you will need:
1. **Python** version 3.10 or higher.
2. An active **OpenRouter** API key (for LLM model access).

---

## 🚀 Step-by-Step Deployment Guide

### Step 1. Clone the Repository and Enter the Directory
Clone the repository and navigate to the project root:
```bash
cd "/Users/nikita/Documents/Gemini/CV Agents"
```

### Step 2. Create and Activate a Virtual Environment
Using a Python virtual environment is recommended for dependency isolation:

**On Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 3. Install Required Dependencies
Install all dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 4. Configure Environment Variables
Create a `.env` file in the project root and add your OpenRouter API key:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### Step 5. Application Configuration (Optional)
In `config.json` you can fine-tune the optimization parameters:
- `global.max_steps` — maximum number of CV optimization iterations.
- `global.k` — number of CV variants generated in parallel per iteration.
- `global.target_score` — desired match threshold (0.0 to 1.0) at which optimization terminates early.
- `global.debug` — enable detailed agent request/response logging to `debug.log`.
- `pipeline.*` — model selection and generation temperature for each agent (parser, composer, auditor, critic, scorer).

---

## 📁 Directory Structure and Input Data

Before running, ensure the following files exist:
- **`cv/main_cv.md`** — your base resume in Markdown format (primary source of truth).
- **`cv/header.txt`** — (optional) HTML/CSS style block or CV header that will be prepended to the final output.
- **`cv/`** — you may place additional context files here (e.g., `linkedin_cv.txt` or `extra.txt`); they will be automatically loaded as auxiliary sources of information about your experience.
- **`vacancies/`** — place one or more job postings as `.txt` files (e.g., `vacancies/vacancy1.txt`).
- **`prompts/`** — system prompt templates for agents (`parser.xml`, `composer.xml`, `audit.xml`, `critic.xml`, `scorer.xml`).

---

## 🏃 Running the Application

To start the resume optimization pipeline:
```bash
python3 main.py
```

### What the Script Does During Execution:
1. Cleans old results and prepares the `results/final_results/` directory.
2. Scans the `vacancies/` directory for job postings.
3. Extracts vacancy requirements into structured JSON format (Parser Agent).
4. Generates $K$ resume variants in parallel using different models (Composer Agent).
5. Validates generated variants for factual accuracy and hallucination-free content (Auditor Agent).
6. Scores valid candidates for relevance against the vacancy (Scorer Agent).
7. If the best score has not reached the target, the Critic Agent analyzes shortcomings and formulates a correction plan for the next iteration.
8. Saves the final resume with the highest match score.

---

## 💾 Results

After execution, optimized resumes are saved to:
`results/final_results/`

The filename format is: `{vacancy_name}_{match_score_percent}.md` (e.g., `backend_developer_96.md`). The output file includes HTML/CSS styling and is ready for PDF export or submission to employers.

---

## 🔍 Debugging and Logs

- If `global.debug` in `config.json` is set to `true`, detailed logs of all model requests and responses are written to `debug.log`. This helps analyze agent reasoning when troubleshooting.