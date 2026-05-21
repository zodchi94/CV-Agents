import os
import glob
import logging
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from schemas import VacancyRequirements, MirrorResult, FinalResult, CompanyInfoResult, CompanyExtraInfo

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Suppress library logs
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.ERROR)

# Load config
load_dotenv()

class VacancyProcessor:
    def __init__(self, model_name: str = 'gemini-flash-lite-latest', max_retries: int = 5, min_score: float = 95.0):
        self.client = genai.Client()
        self.model_name = model_name
        self.max_retries = max_retries
        self.min_score = min_score

    def _call_ai(self, prompt: str, schema: BaseModel):
        return self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.0
            )
        )

    def process_vacancy(self, vacancy_text: str, template: str, mirror_template: str, search_template: str, search_mirror_template: str) -> str:
        # 1. Parse vacancy
        logger.info("Bot 1: Parsing vacancy...")
        response = self._call_ai(template.format(vacancy_text=vacancy_text), VacancyRequirements)
        vacancy_data = VacancyRequirements.model_validate_json(response.text)

        # 2. Mirror vacancy
        current_vacancy = response.text
        for i in range(self.max_retries):
            mirrored_response = self._call_ai(mirror_template.format(input_data=vacancy_text, output_data=current_vacancy), MirrorResult)
            result_obj = MirrorResult.model_validate_json(mirrored_response.text)
            logger.info(f"Vacancy Mirroring Attempt {i+1} Score: {result_obj.alignment_score}%")
            if result_obj.alignment_score >= self.min_score:
                vacancy_data = result_obj.data
                break
            current_vacancy = result_obj.data.model_dump_json()
        
        # 3. Search company info
        company_name = vacancy_data.company.name
        logger.info(f"Bot 2: Searching info for {company_name}...")
        search_result_text = f"Simulated internet search result for {company_name}..." 
        
        # Parse search result
        search_response = self._call_ai(search_template.format(company_name=company_name, search_data=search_result_text), CompanyExtraInfo)
        
        # Mirror search result
        current_search = search_response.text
        company_info = CompanyExtraInfo.model_validate_json(search_response.text)
        for i in range(self.max_retries):
            mirrored_search = self._call_ai(search_mirror_template.format(input_data=search_result_text, output_data=current_search), CompanyInfoResult)
            search_obj = CompanyInfoResult.model_validate_json(mirrored_search.text)
            logger.info(f"Company Info Mirroring Attempt {i+1} Score: {search_obj.alignment_score}%")
            if search_obj.alignment_score >= self.min_score:
                company_info = search_obj.data
                break
            current_search = search_obj.data.model_dump_json()

        # 4. Final aggregation
        final = FinalResult(vacancy=vacancy_data, extra_company_info=company_info)
        return final.model_dump_json(indent=2)

def main():
    VACANCIES_DIR = "vacancies"
    RESULTS_DIR = "results"
    
    with open("prompt_template.xml", 'r') as f: template = f.read()
    with open("mirror_prompt.xml", 'r') as f: mirror_template = f.read()
    with open("search_prompt.xml", 'r') as f: search_template = f.read()
    with open("search_mirror.xml", 'r') as f: search_mirror_template = f.read()

    processor = VacancyProcessor()

    os.makedirs(VACANCIES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    for file_path in glob.glob(os.path.join(VACANCIES_DIR, "*.txt")):
        filename = os.path.basename(file_path)
        with open(file_path, 'r') as f: text = f.read()
        try:
            result = processor.process_vacancy(text, template, mirror_template, search_template, search_mirror_template)
            with open(os.path.join(RESULTS_DIR, filename), 'w') as f: f.write(result)
        except Exception as e:
            logger.error(f"Failed {filename}: {e}")

if __name__ == "__main__":
    main()
