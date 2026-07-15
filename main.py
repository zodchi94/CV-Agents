import os
import json
import glob
import shutil
import asyncio
from pathlib import Path
from schemas import ParserSchema, CVSchema, AuditSchema, PlanSchema
from utils import load_prompt, load_file, calculate_relevance, run_agent, log_debug, SchemaValidationError


def load_config() -> dict:
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def clean_and_create_dirs():
    dirs = [
        "results/final_results"
    ]
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)


async def main():
    # Load configuration
    config = load_config()
    global_cfg = config["global"]
    pipeline = config["pipeline"]

    # Unpack global parameters
    MAX_STEPS: int = global_cfg["max_steps"]
    MAX_PATIENCE: int = global_cfg["max_patience"]
    K: int = global_cfg["k"]
    EPSILON: float = global_cfg["epsilon"]
    TARGET_SCORE: float = global_cfg["target_score"]

    # Initialize results directories
    clean_and_create_dirs()

    # 1. Initialize base data (load from cv/ folder)
    # -------------------------------------------------------
    print("Parsing base CV from cv/main_cv.md and collecting additional information from other files in cv/", end=" ")
    try:
        base_cv: str = load_file("cv/main_cv.md")

        extra_info_parts: list[str] = []
        cv_dir = Path("cv")
        extra_info_parts = [
            f"{p.name}:\n{p.read_text(encoding='utf-8')}"
            for p in cv_dir.iterdir()
            if p.is_file() and p.stem not in ("main_cv", "header")
        ] if cv_dir.is_dir() else []
        extra_info_parts.append("VERIFIED BASE CV CONTENT:\n" + base_cv)
        extra_info = "\n\n".join(extra_info_parts)
        del extra_info_parts
        print("✅")
    except Exception as e:
        print(f"CV parsing failed ❌. Error: {e}")
        raise RuntimeError(f"Failed to load base CV data from cv/ folder: {e}") from e

    # 2. Outer loop: iterate over the vacancy list
    # -------------------------------------------------------
    vacancies_paths = glob.glob("vacancies/*.txt")
    print(f"Found {len(vacancies_paths)} vacancies for processing: {vacancies_paths}")

    for vacancy_path in vacancies_paths:
        vacancy_name = os.path.basename(vacancy_path).split('.')[0]
        print(f"🔹 Processing vacancy: {vacancy_name} ({vacancy_path}): ")
        log_debug(f"=== STARTING VACANCY PROCESSING: {vacancy_name} ({vacancy_path}) ===")
        raw_vacancy = load_file(vacancy_path)

        # 3.1 Extract vacancy requirements into structured JSON
        # ---------------------------------------------------
        print("\tParsing vacancy into structured JSON ", end=" ")
        try:
            _, vacancy_json = await run_agent(
                prompt=load_prompt("prompts/parser.xml"),
                schema=ParserSchema,
                model=pipeline["parser"]["model"],
                temperature=pipeline["parser"]["temperature"],
                vacancy_text=raw_vacancy
            )
            print("✅")
        except SchemaValidationError as e:
            print(f"❌ Parser schema validation failed: {e}")
            log_debug(f"[PARSER SCHEMA ERROR] {e}")
            # Use the raw parsed dict as fallback — downstream code handles missing keys gracefully
            vacancy_json = e.parsed_data
        log_debug(f"Parsed vacancy JSON:\n{json.dumps(vacancy_json, ensure_ascii=False, indent=2)}")

        # 3.2 Start the CV rewriting loop
        # ---------------------------------------------------
        current_cv: str = base_cv
        best_score: float = 0.0
        patience: int = 0
        action_plan: str = "No previous corrections yet. Focus on aligning key skills, technologies, and achievements."
        temperature: float = pipeline["composer"]["temperature"]
        i: int = 0

        # Models assigned to each parallel composer instance (cycle if fewer models than K)
        print("\tStarting CV generation loop: ")
        composer_models = pipeline["composer"]["models"]
        
        while i < MAX_STEPS:
            iter_msg = f"🔸 Iteration {i+1}/{MAX_STEPS} (Temperature: {temperature}, Best score: {best_score:.2f})"
            print(f"\t{iter_msg}")
            log_debug(f"=== {iter_msg} ===")
            
            # 4.2.1. Composer Agent — parallel generation across different models
            print(f"\t\t Generating {K} candidate CVs in parallel using models: \n\t\t {composer_models}...", end=" ")
            composer_prompt = load_prompt("prompts/composer.xml")
            composer_tasks = [
                run_agent(
                    prompt=composer_prompt,
                    schema=CVSchema,
                    model=composer_models[j % len(composer_models)],
                    temperature=temperature,
                    base_cv=current_cv, 
                    extra_info=extra_info, 
                    vacancy=vacancy_json, 
                    action_plan=action_plan
                ) for j in range(K)
            ]

            # Gather results; replace SchemaValidationError with None so one bad
            # candidate doesn't abort the whole batch.
            raw_results = await asyncio.gather(*composer_tasks, return_exceptions=True)
            candidates_raw = []
            for j, res in enumerate(raw_results):
                if isinstance(res, SchemaValidationError):
                    print(f"\n\t\t\t ⚠️  Composer [{j}] schema error: {res}")
                    log_debug(f"[COMPOSER SCHEMA ERROR idx={j}] {res}")
                    # Skip this candidate — it will simply not appear in the pool
                elif isinstance(res, Exception):
                    print(f"\n\t\t\t ⚠️  Composer [{j}] unexpected error: {res}")
                    log_debug(f"[COMPOSER ERROR idx={j}] {res}")
                else:
                    candidates_raw.append(res)

            candidates = [(model, data) for model, data in candidates_raw]
            print("✅" if candidates else " ⚠️  No valid composer candidates this round.")

            if not candidates:
                temperature = round(temperature + 0.1, 1)
                print(f"\t\t All candidates failed validation. Raising temperature to {temperature} ⚠️")
                if temperature > 1.0:
                    print("Temperature limit exceeded. Aborting CV generation loop. ❌")
                    break
                continue

            # 4.2.1. Auditor Agent — factual verification
            print(f"\t\t Running async audit using model ({pipeline['auditor']['model']})...", end=" ")
            audit_prompt = load_prompt("prompts/audit.xml") 
            audit_tasks = [
                run_agent(
                    prompt=audit_prompt,
                    schema=AuditSchema,
                    model=pipeline["auditor"]["model"],
                    temperature=pipeline["auditor"]["temperature"], 
                    candidate_cv=candidate_data['cv_text'], 
                    extra_info=extra_info
                ) for _, candidate_data in candidates
            ]
            raw_audit_results = await asyncio.gather(*audit_tasks, return_exceptions=True)
            evaluations = []
            for idx, res in enumerate(raw_audit_results):
                if isinstance(res, SchemaValidationError):
                    print(f"\n\t\t\t ⚠️  Auditor [{idx}] schema error: {res}")
                    log_debug(f"[AUDITOR SCHEMA ERROR idx={idx}] {res}")
                    # Treat as invalid candidate with empty correction
                    evaluations.append({"is_valid": 0, "c_correction": f"Audit schema error: {res}"})
                elif isinstance(res, Exception):
                    print(f"\n\t\t\t ⚠️  Auditor [{idx}] unexpected error: {res}")
                    log_debug(f"[AUDITOR ERROR idx={idx}] {res}")
                    evaluations.append({"is_valid": 0, "c_correction": f"Audit error: {res}"})
                else:
                    _, eval_data = res
                    evaluations.append(eval_data)
            print("✅")

            # Merge each candidate with its audit evaluation
            audited_candidates = []
            for idx, eval_result in enumerate(evaluations):
                model_name, cv_data = candidates[idx]
                audited_candidates.append({
                    "model": model_name,
                    "cv": cv_data.get('cv_text', ''),
                    "is_valid": eval_result.get('is_valid', 0),
                    "c_correction": eval_result.get('c_correction', '')
                })
            _d = {c['model']: c['is_valid'] for c in audited_candidates}
            print(f"\t\t Audit results: \n\t\t\t {_d}")
            log_debug(f"Audit results: {_d}")
            valid_candidates = [c for c in audited_candidates if c['is_valid'] != 0]

            # Increase temperature if no valid candidates remain
            print(f"\t\t Audit complete: {len(valid_candidates)}/{K} candidates valid {'✅' if valid_candidates else '⚠️'}")
            if not valid_candidates:
                temperature = round(temperature + 0.1, 1)
                print(f"\t\t Raising temperature to {temperature} ⚠️")
                if temperature > 1.0:
                    print("Temperature limit exceeded. Aborting CV generation loop. ❌")
                    break
                continue
                
            # 4.2.2. Scorer Agent — relevance evaluation
            print(f"\t\t Scoring generated CVs using {pipeline['scorer']['model']}...")
            for candidate in valid_candidates:
                candidate['r_score'] = await calculate_relevance(candidate['cv'], vacancy_json)
                print(f"\t\t\t Candidate {candidate['model']}: score = {candidate['r_score']:.2f}")
                log_debug(f"Candidate {candidate['model']} score: {candidate['r_score']:.2f}")

            # Select the iteration winner
            winner = max(valid_candidates, key=lambda x: x['r_score'])
            current_score = winner['r_score']
            print(f"\t\t\t Iteration winner: {winner['model']}  - {winner['r_score']:.2f}  ✅")
            log_debug(f"Iteration winner: {winner['model']} with score {current_score:.2f}")

            
            # 4.2.3. Stopping criteria check
            delta = current_score - best_score

            if current_score >= TARGET_SCORE:
                print(f"\t Target score reached ({current_score:.2f} >= {TARGET_SCORE})! Stopping optimization.   ✅")
                current_cv = winner['cv']
                best_score = current_score
                break

            if delta <= EPSILON:
                patience += 1
                print(f"\t\t Metric improvement negligible (delta = {delta:.2f} <= {EPSILON}). Patience: {patience}/{MAX_PATIENCE} ⚠️")
                if patience >= MAX_PATIENCE:
                    print("\t\t Patience exhausted. Stopping optimization. ⚠️")
                    if current_score > best_score:
                        best_score = current_score
                        current_cv = winner['cv']
                    break
            else:
                print(f"\t\t Metric improved from {best_score:.2f} to {current_score:.2f}! ✅")
                best_score = current_score
                current_cv = winner['cv']
                patience = 0  # Reset patience counter
                
            
            # 4.2.4. Meta-Critic
            print(f"\t\t Running meta-critic ({pipeline['critic']['model']}) for reflection and planning...")
            critic_prompt = load_prompt("prompts/critic.xml")

            # Collect audit corrections from all candidates (including invalidated ones) for the critic
            correction_logs = [c.get('c_correction', '') for c in audited_candidates if c.get('c_correction', '').strip()]
            aggregated_errors = "\n".join(correction_logs) if correction_logs else "No factual errors found in this iteration."
            
            try:
                _, action_plan_response = await run_agent(
                    prompt=critic_prompt,
                    schema=PlanSchema,
                    model=pipeline["critic"]["model"],
                    temperature=pipeline["critic"]["temperature"],
                    current_cv=current_cv,
                    vacancy=vacancy_json,
                    errors=aggregated_errors
                )
                action_plan = action_plan_response.get('action_plan', action_plan)
            except SchemaValidationError as e:
                print(f"\t\t\t ⚠️  Critic schema error: {e}. Keeping previous action plan.")
                log_debug(f"[CRITIC SCHEMA ERROR] {e}")
                # Keep the previous action_plan — the loop continues with the last known good plan
            except Exception as e:
                print(f"\t\t\t ⚠️  Critic error: {e}. Keeping previous action plan.")
                log_debug(f"[CRITIC ERROR] {e}")
                # Keep the previous action_plan — the loop continues with the last known good plan
            print(f"\t\t New correction plan generated: ✅")

            # Increment iteration counter
            i += 1

        # 4. Save final result for the current vacancy
        print(f"\t Saving final result for {vacancy_name} with best score {best_score:.2f}...")

        final_file_path = f"results/final_results/{vacancy_name}_{int(best_score * 100)}.md"
        try:
            HEADER = load_file("cv/header.txt")
        except Exception:
            HEADER = (
                "<div>\n"
                "<style>\n"
                "  body, p, li {font-size: 12px !important;}\n"
                "  h2 {font-size: 23px !important;}\n"
                "  h3 {font-size: 14px !important;}\n"
                "</style>\n\n"
            )
    
        with open(final_file_path, "w", encoding="utf-8") as f:
            f.write(HEADER + current_cv + "</div>\n\n")

        print(f"\t Saved: {final_file_path}")

if __name__ == "__main__":
    asyncio.run(main())
