import os
import glob
import logging
import warnings
from dotenv import load_dotenv
from google import genai
from schemas import VacancyRequirements, MirrorResult, FinalResult, CompanyInfoResult, CompanyExtraInfo

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
warnings.filterwarnings("ignore")
for log_name in ["httpx", "google", "google.genai", "urllib3"]:
    logging.getLogger(log_name).setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

load_dotenv()

class BaseAgent:
    def __init__(self, model_name: str = 'gemini-flash-lite-latest'):
        self.client = genai.Client()
        self.model_name = model_name

    def call_ai(self, prompt: str, schema: BaseModel):
        return self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.0
            )
        )

class VacancyProcessor(BaseAgent):
    def __init__(self, prompts_dir: str = "prompts", min_score: float = 95.0, **kwargs):
        super().__init__(**kwargs)
        self.prompts_dir = prompts_dir
        self.min_score = min_score

    def _get_prompt(self, filename: str) -> str:
        with open(os.path.join(self.prompts_dir, filename), 'r') as f: return f.read()

    def process_all(self, vacancies_dir: str, results_dir: str):
        t = self._get_prompt("prompt_template.xml")
        m = self._get_prompt("mirror_prompt.xml")
        s = self._get_prompt("search_prompt.xml")
        sm = self._get_prompt("search_mirror.xml")

        for file_path in glob.glob(os.path.join(vacancies_dir, "*.txt")):
            with open(file_path, 'r') as f: text = f.read()
            
            # Bot 1: Vacancy Parsing & Mirroring
            logger.info(f"Bot 1: Processing {os.path.basename(file_path)}...")
            res1 = self.call_ai(t.format(vacancy_text=text), VacancyRequirements)
            data = VacancyRequirements.model_validate_json(res1.text)
            
            # Mirroring logic (simplified)
            mirrored = self.call_ai(m.format(input_data=text, output_data=res1.text), MirrorResult)
            data = MirrorResult.model_validate_json(mirrored.text).data
            logger.info(f"Alignment Score: {MirrorResult.model_validate_json(mirrored.text).alignment_score}%")

            # Bot 2: Search Company Info
            logger.info(f"Bot 2: Searching {data.company.name}...")
            search_res = self.call_ai(s.format(company_name=data.company.name, search_data="Simulated search..."), CompanyExtraInfo)
            
            # Mirror search
            mirrored_search = self.call_ai(sm.format(input_data="...", output_data=search_res.text), CompanyInfoResult)
            info = CompanyInfoResult.model_validate_json(mirrored_search.text).data

            final = FinalResult(vacancy=data, extra_company_info=info)
            with open(os.path.join(results_dir, os.path.basename(file_path)), 'w') as f: f.write(final.model_dump_json(indent=2))
