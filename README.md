# CV Agent System

## Setup
1. Create a `.env` file with `GEMINI_API_KEY`.
2. Place vacancy files in `vacancies/`.
3. Place your CV in `cv/my_cv.txt`.
4. Install dependencies: `pip3 install python-dotenv google-genai pydantic`.

## Usage
1. Parse vacancies: `python3 vacancy_agent.py`
2. Get advice: `python3 advisor.py`
