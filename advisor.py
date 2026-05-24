import os
import logging
import warnings
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
warnings.filterwarnings("ignore")
for log_name in ["httpx", "google", "google.genai", "urllib3"]:
    logging.getLogger(log_name).setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

load_dotenv()

class AdvisorBot:
    def __init__(self):
        self.client = genai.Client()
        self.model = 'gemini-flash-lite-latest'

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        reader = PdfReader(pdf_path)
        text = "".join([page.extract_text() for page in reader.pages])
        return text

    def get_recommendations(self, vacancy_json: str, cv_text: str):
        prompt = f"""
        Compare the following job vacancy and CV. Provide specific, actionable recommendations on what should be improved in the CV to increase matching chances.

        Vacancy:
        {vacancy_json}

        CV:
        {cv_text}
        """
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        return response.text

    def rewrite_cv(self, cv_text: str, recommendations: str):
        prompt = f"""
        Given the original CV and recommendations for improvement, rewrite the CV to perfectly match the job requirements. Keep the original formatting and style.

        Original CV:
        {cv_text}

        Recommendations:
        {recommendations}

        Output the full rewritten CV in text format.
        """
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        return response.text

def main():
    RESULTS_DIR = "results"
    CV_PATH = "cv/Machine Learning Manager_Nikita_Kudriashov.pdf"
    
    bot = AdvisorBot()
    cv_text = bot.extract_text_from_pdf(CV_PATH)

    for vacancy_file in os.listdir(RESULTS_DIR):
        if not vacancy_file.endswith(".txt"): continue
            
        with open(os.path.join(RESULTS_DIR, vacancy_file), 'r') as f: vacancy_json = f.read()
        
        logger.info(f"Generating and applying recommendations for {vacancy_file}...")
        recs = bot.get_recommendations(vacancy_json, cv_text)
        new_cv = bot.rewrite_cv(cv_text, recs)
        
        output_name = f"cv_{vacancy_file.replace('.txt', '.txt')}"
        with open(os.path.join("cv", output_name), 'w') as f: f.write(new_cv)
        logger.info(f"Rewritten CV saved to cv/{output_name}")

if __name__ == "__main__":
    main()
