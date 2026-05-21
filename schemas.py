from pydantic import BaseModel, Field
from typing import List, Optional

class Company(BaseModel):
    name: str = Field(description="Name of the company. If not mentioned, return 'Not specified'")
    domain: str = Field(description="Main industry or business area. If unknown, return 'Unknown'")

class Role(BaseModel):
    level: str = Field(description="Seniority level. If unknown, return 'Not specified'")
    is_people_manager: bool = Field(description="Does the role involve direct people management/HR duties?")
    is_tech_leader: bool = Field(description="Does the role involve technical leadership?")
    tasks: List[str] = Field(description="Core responsibilities and task types")
    team: str = Field(description="Name of the specific team. If unknown, return 'Not specified'")
    job_focus: str = Field(description="Main focus of the job (e.g., RnD, Production, Applied ML)")
    project: str = Field(description="Name of the product or project")
    role_domain: str = Field(description="Specific business or problem domain the role focuses on")

class AIFocus(BaseModel):
    requires_classical_ml: bool = Field(description="Does the role require classical ML?")
    requires_llms: bool = Field(description="Does the role require Generative AI / LLM usage?")
    requires_agentic_ai: bool = Field(description="Does the role explicitly require Agentic AI / Autonomous Agents?")
    extra: bool = Field(description="Does the role contain some extra AI requirements (e.g., RecSys, Antifraud)?")

class Skill(BaseModel):
    skill_name: str = Field(description="Name of the skill or technology")
    critical_level: str = Field(description="How critical this skill is (Core, Secondary, Optional)")
    proficiency_level: str = Field(description="Required candidate proficiency level (Low, Middle, High)")
    context: str = Field(description="How the skill will be used in practice")

class VacancyRequirements(BaseModel):
    company: Company = Field(description="Information about the company")
    role: Role = Field(description="General description of the role")
    ai_focus: AIFocus = Field(description="Explicit check for AI paradigms")
    tech_skills: List[Skill] = Field(description="Technical skills required")
    soft_skills: List[Skill] = Field(description="Soft skills required")

class CompanyExtraInfo(BaseModel):
    info: str = Field(description="Useful gathered info about the company for resume tailoring")

class CompanyInfoResult(BaseModel):
    data: CompanyExtraInfo = Field(description="The gathered company info")
    alignment_score: float = Field(description="Accuracy score of the internet search parsing")

class MirrorResult(BaseModel):
    data: VacancyRequirements = Field(description="The corrected/mirrored vacancy data")
    alignment_score: float = Field(description="Percentage (0-100) of how well the output matches the original input.")

class FinalResult(BaseModel):
    vacancy: VacancyRequirements = Field(description="The original vacancy requirements")
    extra_company_info: CompanyExtraInfo = Field(description="Extra information about the company")
