import glob
from agent_framework import Agent
from schemas import (
    VacancyRequirements, MirrorResult, DraftCV, 
    CompanyExtraInfo, FinalResult, RefinementResult
)
PROMPTS_PATH = "prompts/"

def run_pipeline(vacancy_path: str, cv_path: str):
    with open(vacancy_path, 'r') as f:
        vacancy_text = f.read()
    with open(cv_path, 'r') as f:
        cv_text = f.read()

    # 1. Vacancy Parsing
    parser_agent = Agent(f"{PROMPTS_PATH}vacancy_parser.xml")
    vacancy_reqs = parser_agent.run(VacancyRequirements, {"vacancy_text": vacancy_text})
    
    # 2. Vacancy Parsing Mirroring
    mirrored_score, cnt = 0, 0
    while mirrored_score < 100 | cnt < 5:
        mirror_agent = Agent(f"{PROMPTS_PATH}vacancy_mirror.xml")
        mirrored = mirror_agent.run(MirrorResult, {"input_data": vacancy_text, "output_data": vacancy_reqs.model_dump_json()})
        vacancy_reqs, mirrored_score = mirrored.data, mirrored.alignment_score
        print(f"Mirroring attempt {cnt+1}: Alignment Score = {mirrored_score}%")
        cnt+=1

    # 3. Research
    researcher_agent = Agent(f"{PROMPTS_PATH}web_searcher.xml")
    research = researcher_agent.run(CompanyExtraInfo, {"company_name": vacancy_reqs.company.name})
    vacancy_reqs_extended = FinalResult(vacancy=vacancy_reqs, extra_company_info=research)

    # 4. & 5. Draft & Refine CV Loop
    draft_cv_agent = Agent(f"{PROMPTS_PATH}cv_draft_builder.xml")
    refinement_agent = Agent(f"{PROMPTS_PATH}cv_refiner.xml")
    
    current_cv_text = cv_text
    current_corrections = ""
    refined = None
    
    # Static styling and header template
    header_template = (
        "<div>\n"
        "<style>\n"
        "  body, p, li {{font-size: 12px !important;}}\n"
        "  h2 {{font-size: 23px !important;}}\n"
        "  h3 {{font-size: 14px !important;}}\n"
        "</style>\n\n"
        "## Nikita Kudriashov ( {professional_title} ) \n\n"
        "Serbia | 0637238966 | [nskudriashov@gmail.com](mailto:nskudriashov@gmail.com) | "
        "[linkedin.com/in/nkudriashov](http://linkedin.com/in/nkudriashov) | "
        "[kaggle.com/nikitakudriashov](http://kaggle.com/nikitakudriashov) | "
        "[github.com/zodchi94](https://github.com/zodchi94)\n\n"
    )

    for i in range(5):
        draft = draft_cv_agent.run(DraftCV, {
            "vacancy_data": vacancy_reqs_extended, 
            "cv_text": current_cv_text,
            "corrections": current_corrections
        })
        
        # Build complete CV including professional header and styling
        complete_cv_header = header_template.format(professional_title=draft.professional_title)
        complete_cv_text = complete_cv_header + draft.content

        refined = refinement_agent.run(RefinementResult, {
            "vacancy_data": vacancy_reqs, 
            "current_cv": complete_cv_text, 
            "company_info": research.info
        })
        
        print(f"Refinement iteration {i+1}: Score = {refined.alignment_score}%")
        
        if refined.alignment_score >= 100:
            break
            
        current_cv_text = refined.data.content
        current_corrections = refined.improvement_notes

    return vacancy_reqs_extended, refined

def main():
    # Example usage
    for file_path in glob.glob("vacancies/*.txt"):
        res, refined_cv = run_pipeline(file_path, "cv/main_cv.md")
        print(f"Processed {file_path}")

if __name__ == "__main__":
    main()
