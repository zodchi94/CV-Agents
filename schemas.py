from pydantic import BaseModel, Field
from typing import List, Optional

class Company(BaseModel):
    company_name: str = Field(description="Name of the company. If not mentioned, return 'Not specified'")
    company_domain: str = Field(description="Main industry or business area. If unknown, return 'Unknown'")
    business_scale: Optional[str] = Field(default=None, description="Business scale if mentioned")

class Role(BaseModel):
    seniority_level: str = Field(description="Seniority level. If unknown, return 'Not specified'")
    is_people_manager: bool = Field(description="Does the role involve direct people management/HR duties?")
    is_tech_leader: bool = Field(description="Does the role involve technical leadership?")
    team_name: str = Field(description="Name of the specific team. If unknown, return 'Not specified'")
    job_focus: str = Field(description="Main focus of the job (e.g., RnD, Production, Applied ML)")
    project_name: str = Field(description="Name of the product or project")
    role_domain: str = Field(description="Specific business or problem domain the role focuses on")
    data_signals: List[str] = Field(description="Specific types of data or features the role works with")
    operational_constraints: List[str] = Field(description="Production/operational requirements and constraints")
    planning_horizon: Optional[str] = Field(default=None, description="Architectural or strategic timeframes")

class AIFocus(BaseModel):
    requires_classical_ml: bool = Field(description="Does the role require classical ML?")
    requires_llms: bool = Field(description="Does the role require Generative AI / LLM usage?")
    requires_agentic_ai: bool = Field(description="Does the role explicitly require Agentic AI / Autonomous Agents?")
    extra: List[str] = Field(description="List of extra AI/ML subfields or requirements")

class VacancyRequirements(BaseModel):
    company_and_domain: Company = Field(description="Information about the company and domain")
    role_and_architecture: Role = Field(description="General description of the role and architecture")
    tasks_and_responsibilities: List[str] = Field(description="Extracted tasks and responsibilities")
    ai_and_ml_methodology_focus: AIFocus = Field(description="Explicit check for AI paradigms")
    skills_analysis: dict = Field(description="Parsed technical/soft skills and language requirements")

class ParserSchema(VacancyRequirements):
    pass

class CVSchema(BaseModel):
    cv_text: str = Field(description="The complete content of the tailored resume in Markdown format")

class AuditSchema(BaseModel):
    is_valid: int = Field(description="1 if the CV is strictly factual and matches extra_info/base_cv with absolutely zero hallucinations or unverified facts. 0 if it contains any hallucinated info, false claims, or unverified facts.")
    c_correction: str = Field(description="Detailed corrections or bullet points highlighting exactly what needs to be fixed or removed. Empty if is_valid is 1.")

class PlanSchema(BaseModel):
    action_plan: str = Field(description="A detailed action plan outlining what specific parts of the CV need to be modified or improved next.")

class RelevanceResponse(BaseModel):
    score: float = Field(description="Relevance score from 0.0 to 1.0. 1.0 means perfect alignment, 0.0 means completely irrelevant.")
