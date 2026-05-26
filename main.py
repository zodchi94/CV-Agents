import glob
import os
from agent_framework import Agent
from schemas import (
    VacancyRequirements, MirrorResult, DraftCV, 
    CompanyExtraInfo, FinalResult, RefinementResult
)
PROMPTS_PATH = "prompts/"

def load_cv_files():
    cv_data = {}
    for file_path in glob.glob("cv/*"):
        name = os.path.basename(file_path).split('.')[0]
        with open(file_path, 'r') as f:
            cv_data[name] = f.read()
    return cv_data

def run_pipeline(vacancy_path: str):
    with open(vacancy_path, 'r') as f:
        vacancy_text = f.read()
    
    cv_files = load_cv_files()
    
    # 1. Vacancy Parsing
    print("Step 1: Parsing Vacancy Requirements...")
    parser_agent = Agent(f"{PROMPTS_PATH}vacancy_parser.xml")
    vacancy_reqs = parser_agent.run(VacancyRequirements, {"vacancy_text": vacancy_text})
    
    # 2. Vacancy Parsing Mirroring
    print("Step 2: Mirroring Vacancy Requirements...")
    mirrored_score, cnt = 0, 0
    while mirrored_score < 98 and cnt < 3:
        mirror_agent = Agent(f"{PROMPTS_PATH}vacancy_mirror.xml")
        mirrored = mirror_agent.run(MirrorResult, {"input_data": vacancy_text, "output_data": vacancy_reqs.model_dump_json()})
        vacancy_reqs, mirrored_score = mirrored.data, mirrored.alignment_score
        print(f"Mirroring attempt {cnt+1}: Alignment Score = {mirrored_score}%")
        cnt+=1
    
    # 3. Research
    print("Step 3: Researching Company Information...")
    researcher_agent = Agent(f"{PROMPTS_PATH}web_searcher.xml")
    research = researcher_agent.run(CompanyExtraInfo, {"company_name": vacancy_reqs.company.name})
    vacancy_reqs_extended = FinalResult(vacancy=vacancy_reqs, extra_company_info=research)

    # 4. & 5. Draft & Refine CV Loop
    print("Step 4 & 5: Drafting and Refining CV...")
    draft_cv_agent = Agent(f"{PROMPTS_PATH}cv_draft_builder.xml")
    refinement_agent = Agent(f"{PROMPTS_PATH}cv_refiner.xml")
    
    accumulated_corrections = ""
    refined = None
    alignment_score, cnt = 0, 0
    
    while alignment_score < 98 and cnt < 3:
        # Build from scratch with original source files + accumulated corrections
        draft = draft_cv_agent.run(DraftCV, {
            "vacancy_data": vacancy_reqs_extended, 
            "cv_text": str(cv_files),
            "corrections": accumulated_corrections
        })
        
        refined = refinement_agent.run(RefinementResult, {
            "vacancy_data": vacancy_reqs, 
            "current_cv": draft.content, 
            "company_info": research.info
        })
        
        alignment_score = refined.alignment_score
        print(f"Refinement iteration {cnt+1}: Score = {alignment_score}%")
            
        accumulated_corrections += f"\nIteration {cnt+1} feedback: {refined.improvement_notes}"
        cnt += 1

    # 6. Final Human/Balance Refinement
    print("Step 6: Final Human/Balance Refinement...")
    human_refiner_agent = Agent(f"{PROMPTS_PATH}human_refiner.xml")
    final_refined_cv = human_refiner_agent.run(DraftCV, {
        "tailored_cv": refined.data.content,
        "vacancy_data": vacancy_reqs.model_dump_json()
    })

    # Save final refined CV to results folder
    os.makedirs("results", exist_ok=True)
    company_filename = vacancy_reqs.company.name.lower().replace(" ", "_")
    
    header = (
        "<div>\n"
        "<style>\n"
        "  body, p, li {font-size: 12px !important;}\n"
        "  h2 {font-size: 23px !important;}\n"
        "  h3 {font-size: 14px !important;}\n"
        "</style>\n\n"
        f"## Nikita Kudriashov ( {final_refined_cv.professional_title} ) \n\n"
        "Serbia | 0637238966 | [nskudriashov@gmail.com](mailto:nskudriashov@gmail.com) | "
        "[linkedin.com/in/nkudriashov](http://linkedin.com/in/nkudriashov) | "
        "[kaggle.com/nikitakudriashov](http://kaggle.com/nikitakudriashov) | "
        "[github.com/zodchi94](https://github.com/zodchi94)\n\n"
        "</div>\n\n"
    )

    with open(f"results/{company_filename}.md", "w") as f:
        f.write(header + final_refined_cv.content)

    return vacancy_reqs_extended, refined

def main():
    # Example usage
    for file_path in glob.glob("vacancies/*.txt"):
        print(f"Processing {file_path}")
        run_pipeline(file_path)
        
        
if __name__ == "__main__":
    main()
