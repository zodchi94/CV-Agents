from vacancy_agent import BaseAgent
from hr_schema import HRScore

class HRAgent(BaseAgent):
    def evaluate(self, resume: str, vacancy: str) -> HRScore:
        prompt = f"""
        Evaluate how well this resume matches the vacancy.
        Vacancy: {vacancy}
        Resume: {resume}
        
        Provide a score from 0 to 10 and brief reasoning.
        """
        res = self.call_ai(prompt, HRScore)
        return HRScore.model_validate_json(res.text)
