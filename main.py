import glob
import os
import shutil
from agent_framework import Agent
from schemas import (
    VacancyRequirements, MirrorResult, DraftCV, 
    CompanyExtraInfo, FinalResult, RefinementResult
)
PROMPTS_PATH = "prompts/"

def clean_and_create_dirs():
    dirs = [
        "results/research_jsons",
        "results/draft_cvs",
        "results/refined_cvs",
        "results/final_results"
    ]
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

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
    
    vacancy_name = os.path.basename(vacancy_path).split('.')[0]
    cv_files = load_cv_files()
    
    # 1. Vacancy Parsing
    print("Step 1: Parsing Vacancy Requirements...")
    parser_agent = Agent(f"{PROMPTS_PATH}vacancy_parser.xml")
    vacancy_reqs = parser_agent.run(VacancyRequirements, {"vacancy_text": vacancy_text})
    
    # # 1.1 Vacancy Parsing Mirroring
    # print("Step 1.1: Mirroring Vacancy Requirements...")
    # mirrored_score, cnt = 0, 0
    # while mirrored_score < 98 and cnt < 5:
    #     mirror_agent = Agent(f"{PROMPTS_PATH}vacancy_mirror.xml")
    #     mirrored = mirror_agent.run(MirrorResult, {"input_data": vacancy_text, "output_data": vacancy_reqs.model_dump_json()})
    #     vacancy_reqs, mirrored_score = mirrored.data, mirrored.alignment_score
    #     print(f"Mirroring attempt {cnt+1}: Alignment Score = {mirrored_score}%")
    #     cnt+=1

    # Save intermediate Research JSON
    with open(f"results/research_jsons/{vacancy_name}.json", "w") as f:
        f.write(vacancy_reqs.model_dump_json(indent=2))

    # # 4. & 5. Draft & Refine CV Loop
    # print("Step 4 & 5: Drafting and Refining CV...")
    # draft_cv_agent = Agent(f"{PROMPTS_PATH}cv_draft_builder.xml")
    # refinement_agent = Agent(f"{PROMPTS_PATH}cv_refiner.xml")
    
    # cv_history = f"Base CV source files:\n{str(cv_files)}\n"
    # corrections_history = ""
    # refined = None
    # alignment_score, cnt = 0, 0
    
    # while alignment_score < 98 and cnt < 5:
    #     draft = draft_cv_agent.run(DraftCV, {
    #         "vacancy_data": vacancy_reqs_extended, 
    #         "cv_history": cv_history,
    #         "corrections_history": corrections_history if corrections_history else "No previous corrections yet. Rely on your own expertise."
    #     })
        
    #     if cnt == 0:
    #         with open(f"results/draft_cvs/{vacancy_name}_draft.md", "w") as f:
    #             f.write(f"# Professional Title: {draft.professional_title}\n\n" + draft.content)

    #     refined = refinement_agent.run(RefinementResult, {
    #         "vacancy_data": vacancy_reqs, 
    #         "current_cv": draft.content, 
    #         "company_info": research.info
    #     })
        
    #     alignment_score = refined.alignment_score
    #     print(f"Refinement iteration {cnt+1}: Score = {alignment_score}%")
        
    #     # Accumulate history of drafts and corrections
    #     cv_history += f"\n--- Draft version {cnt+1} ---\nTitle: {draft.professional_title}\nContent:\n{draft.content}\n"
    #     corrections_history += f"\nFeedback on Draft version {cnt+1}:\nScore: {refined.alignment_score}\nNotes: {refined.improvement_notes}\n"
            
    #     cnt += 1
    
    # # Save intermediate Refined CV
    # if refined:
    #     with open(f"results/refined_cvs/{vacancy_name}_refined_{int(alignment_score)}.md", "w") as f:
    #         f.write(f"# Professional Title: {refined.data.professional_title}\n\n" + refined.data.content)

    # # 6. Final Human/Balance Refinement
    # print("Step 6: Final Human/Balance Refinement...")
    # human_refiner_agent = Agent(f"{PROMPTS_PATH}human_refiner.xml")
    # final_cv_content = human_refiner_agent.run(DraftCV, {
    #     "original_cv": str(cv_files),
    #     "tailored_cv": refined.data.content,
    #     "vacancy_description": vacancy_reqs.model_dump_json()
    # })
    
    # header = (
    #     "<div>\n"
    #     "<style>\n"
    #     "  body, p, li {font-size: 12px !important;}\n"
    #     "  h2 {font-size: 23px !important;}\n"
    #     "  h3 {font-size: 14px !important;}\n"
    #     "</style>\n\n"
    #     f"## Nikita Kudriashov ( {final_cv_content.professional_title} ) \n\n"
    #     "Serbia | 0637238966 | [nskudriashov@gmail.com](mailto:nskudriashov@gmail.com) | "
    #     "[linkedin.com/in/nkudriashov](http://linkedin.com/in/nkudriashov) | "
    #     "[kaggle.com/nikitakudriashov](http://kaggle.com/nikitakudriashov) | "
    #     "[github.com/zodchi94](https://github.com/zodchi94)\n\n"
    #     "</div>\n\n"
    # )

    # # Save final results
    # with open(f"results/final_results/{vacancy_name}_{int(alignment_score)}.md", "w") as f:
    #     f.write(header + final_cv_content.content)

    # return vacancy_reqs_extended, refined
    return None

def main():
    # Example usage
    clean_and_create_dirs()
    for file_path in glob.glob("vacancies/*.txt"):
        print(f"Processing {file_path}")
        run_pipeline(file_path)
        
        
if __name__ == "__main__":
    main()
