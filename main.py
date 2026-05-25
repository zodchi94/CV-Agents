import os
import glob
import logging
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from schemas import VacancyRequirements, MirrorResult, CompanyInfoResult, CompanyExtraInfo
from cv_schemas import DraftCV, RefinementResult
from hr_agent import HRAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Suppress library logs
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.ERROR)
# Suppress specific AFC logging
logging.getLogger("google.genai.live").setLevel(logging.ERROR)

# Load config
load_dotenv()
with open("config.json", "r") as f:
    config = json.load(f)

class VacancyProcessor:
    def __init__(self, max_retries: int = 5, min_score: float = 95.0):
        self.client = genai.Client()
        self.max_retries = max_retries
        self.min_score = min_score

    def _call_ai(self, prompt: str, schema: BaseModel, bot_name: str):
        bot_config = config['bots'][bot_name]
        return self.client.models.generate_content(
            model=bot_config['model'],
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=bot_config['temperature']
            )
        )

    def generate_cv(self, vacancy_data: VacancyRequirements, company_info: CompanyExtraInfo, draft_prompt: str, refinement_prompt: str) -> str:
        # Load CV contents
        cv_files = glob.glob(os.path.join("cv", "*"))
        candidate_experience = ""
        for file_path in cv_files:
            with open(file_path, 'r') as f:
                candidate_experience += f"\n--- {os.path.basename(file_path)} ---\n" + f.read()

        # 1. Generate Draft
        logger.info("Bot 3: Generating CV draft...")
        draft_response = self._call_ai(draft_prompt.format(vacancy_data=vacancy_data.model_dump_json(), company_info=company_info.model_dump_json(), cv_data=candidate_experience), DraftCV, 'cv_generator')
        draft_cv = DraftCV.model_validate_json(draft_response.text)

        # 2. Refine
        current_cv = draft_cv.content
        for i in range(self.max_retries):
            refinement_response = self._call_ai(refinement_prompt.format(current_cv=current_cv, vacancy_data=vacancy_data.model_dump_json(), company_info=company_info.model_dump_json()), RefinementResult, 'cv_generator')
            refinement_obj = RefinementResult.model_validate_json(refinement_response.text)
            logger.info(f"CV Refinement Attempt {i+1} Score: {refinement_obj.alignment_score}%")
            if refinement_obj.alignment_score >= self.min_score:
                # 3. HR Evaluation
                logger.info("Bot 4: Evaluating match...")
                hr_agent = HRAgent()
                hr_score = hr_agent.evaluate(candidate_experience, vacancy_data.model_dump_json())
                logger.info(f"HR Assessment Score: {hr_score.score}/10")
                logger.info(f"Reasoning: {hr_score.reasoning}")
                return refinement_obj.data.content
            current_cv = refinement_obj.data.content
            
        return current_cv

    def process_vacancy(self, vacancy_text: str, parser_prompt: str, researcher_prompt: str, draft_prompt: str, refinement_prompt: str) -> str:
        # 1. Parse & Audit vacancy (Combined Bot)
        logger.info("Bot 1: Parsing & auditing vacancy...")
        combined_response = self._call_ai(parser_prompt.format(vacancy_text=vacancy_text), MirrorResult, 'vacancy_parser')
        result_obj = MirrorResult.model_validate_json(combined_response.text)
        vacancy_data = result_obj.data
        logger.info(f"Parsing/Audit Alignment Score: {result_obj.alignment_score}%")

        # 2. Search company info
        company_name = vacancy_data.company.name
        logger.info(f"Bot 2: Searching info for {company_name}...")
        search_result_text = f"Simulated internet search result for {company_name}..." 
        
        # Researcher
        search_response = self._call_ai(researcher_prompt.format(company_name=company_name, search_data=search_result_text), CompanyInfoResult, 'company_researcher')
        company_info = CompanyInfoResult.model_validate_json(search_response.text).data
        logger.info(f"Company Info Alignment Score: {CompanyInfoResult.model_validate_json(search_response.text).alignment_score}%")

        # 3. Generate CV
        return self.generate_cv(vacancy_data, company_info, draft_prompt, refinement_prompt)

def main():
    VACANCIES_DIR = "vacancies"
    RESULTS_DIR = "results"
    PROMPT_DIR = "prompts"
    
    with open(f"{PROMPT_DIR}/vacancy_parser.xml", 'r') as f: parser_prompt = f.read()
    with open(f"{PROMPT_DIR}/company_researcher.xml", 'r') as f: researcher_prompt = f.read()
    with open(f"{PROMPT_DIR}/cv_draft_prompt.xml", 'r') as f: draft_prompt = f.read()
    with open(f"{PROMPT_DIR}/cv_refinement_prompt.xml", 'r') as f: refinement_prompt = f.read()

    processor = VacancyProcessor()

    os.makedirs(VACANCIES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    for file_path in glob.glob(os.path.join(VACANCIES_DIR, "*.txt")):
        filename = os.path.basename(file_path)
        md_filename = os.path.splitext(filename)[0] + ".md"
        with open(file_path, 'r') as f: text = f.read()
        try:
            result = processor.process_vacancy(text, parser_prompt, researcher_prompt, draft_prompt, refinement_prompt)
            with open(os.path.join(RESULTS_DIR, md_filename), 'w') as f: f.write(result)
        except Exception as e:
            logger.error(f"Failed {filename}: {e}")

if __name__ == "__main__":
    main()
