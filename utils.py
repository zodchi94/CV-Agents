import os
import asyncio
from datetime import datetime
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
import httpx
from schemas import RelevanceResponse
import json

load_dotenv()

# Absolute paths to config and log files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DEBUG_LOG_PATH = os.path.join(BASE_DIR, "debug.log")

_config_cache = None


class SchemaValidationError(Exception):
    """Raised when an agent's response fails Pydantic schema validation."""
    def __init__(self, message: str, validation_errors: list, parsed_data: dict):
        super().__init__(message)
        self.validation_errors = validation_errors  # list of pydantic error dicts
        self.parsed_data = parsed_data              # the raw parsed dict that failed


def get_config() -> dict:
    """Get loaded configuration, cached in memory."""
    global _config_cache
    if _config_cache is None:
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                _config_cache = json.load(f)
        except Exception:
            return {}
    return _config_cache


def log_debug(message: str):
    """Log messages to debug.log if debug is enabled in config.json."""
    cfg = get_config()
    debug_enabled = cfg.get("global", {}).get("debug", False)

    if debug_enabled:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {message}\n")
        except Exception:
            pass


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def clean_json_string(text: str) -> str:
    if text:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
    return text


def _escape_control_chars_in_json(text: str) -> str:
    """Escape literal control characters that appear inside JSON string values."""
    result = []
    i = 0
    in_string = False
    escape_next = False
    while i < len(text):
        char = text[i]
        if not in_string:
            if char == '"':
                in_string = True
            result.append(char)
        else:
            if escape_next:
                escape_next = False
                result.append(char)
            elif char == '\\':
                escape_next = True
                result.append(char)
            elif char == '"':
                in_string = False
                result.append(char)
            elif ord(char) < 32:
                result.append(json.dumps(char)[1:-1])
            else:
                result.append(char)
        i += 1
    return ''.join(result)


def safe_json_loads(text: str) -> dict:
    text = clean_json_string(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fixed = _escape_control_chars_in_json(text)
        return json.loads(fixed)


def _inject_schema_into_prompt(prompt: str, schema) -> str:
    """
    Injects the JSON schema of the given Pydantic model into the prompt body.
    The schema block is placed just before the closing </root> tag so it sits
    inside the prompt's root element. If no </root> tag is found, the block is
    appended at the end of the prompt.
    """
    json_schema = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
    schema_block = (
        "\n<output_schema>\n"
        "Your response MUST be a single valid JSON object that strictly conforms to the following schema. "
        "All required fields must be present with the correct types.\n"
        f"{json_schema}\n"
        "</output_schema>\n"
    )
    close_tag = "</root>"
    if close_tag in prompt:
        return prompt.replace(close_tag, schema_block + close_tag, 1)
    return prompt + schema_block


async def run_agent(prompt: str, schema, model: str, temperature: float, **kwargs) -> tuple[str, dict]:
    # Formats keyword arguments safely
    # If a value in kwargs is not a string, let's convert it to string or JSON
    formatted_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, (dict, list, BaseModel)):
            if hasattr(v, "model_dump_json"):
                formatted_kwargs[k] = v.model_dump_json()
            else:
                formatted_kwargs[k] = json.dumps(v, ensure_ascii=False)
        else:
            formatted_kwargs[k] = str(v)
    
    if formatted_kwargs:
        formatted_prompt = prompt.format(**formatted_kwargs)
    else:
        formatted_prompt = prompt

    # Inject the Pydantic schema into the prompt body (inside <root> tags)
    if schema is not None:
        formatted_prompt = _inject_schema_into_prompt(formatted_prompt, schema)
    
    # Retrieve OpenRouter API key
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set. Please set it to run the pipeline.")
    
    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/zodchi94/CV-Agents",
        "X-Title": "CV Agents"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": formatted_prompt
            }
        ],
        "temperature": temperature
    }

    # Force JSON response mode when a schema is provided
    if schema is not None:
        payload["response_format"] = {"type": "json_object"}
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text}")
        
        result_json = response.json()
        if "choices" not in result_json or len(result_json["choices"]) == 0:
            raise RuntimeError(f"Unexpected OpenRouter response format: {json.dumps(result_json)}")
            
        content = result_json["choices"][0]["message"]["content"]

        parsed = safe_json_loads(content)
        
        # Log agent inputs/outputs for debug mode
        log_debug(
            f"=== AGENT RUN ===\n"
            f"Model: {model}\n"
            f"Temperature: {temperature}\n"
            f"Schema: {schema.__name__ if schema else 'None'}\n"
            f"Formatted Prompt:\n{formatted_prompt}\n\n"
            f"Raw Response:\n{content}\n\n"
            f"Parsed JSON:\n{json.dumps(parsed, ensure_ascii=False, indent=2)}\n"
            f"=================\n"
        )

        if schema is not None:
            try:
                validated = schema.model_validate(parsed)
                return model, validated.model_dump()
            except ValidationError as e:
                err_details = e.errors()
                err_msg = (
                    f"Schema validation failed for '{schema.__name__}'. "
                    f"{len(err_details)} error(s): "
                    + "; ".join(
                        f"[{' -> '.join(str(loc) for loc in err['loc'])}] {err['msg']}"
                        for err in err_details
                    )
                )
                print(f"\t\t\t ⚠️  {err_msg}")
                log_debug(
                    f"[VALIDATION ERROR] {err_msg}\n"
                    f"Parsed Data:\n{json.dumps(parsed, ensure_ascii=False, indent=2)}\n"
                )
                raise SchemaValidationError(err_msg, err_details, parsed)

        return model, parsed


async def calculate_relevance(cv_text: str, vacancy_json: dict) -> float:
    # Evaluate CV relevance against vacancy requirements using OpenRouter
    if isinstance(vacancy_json, BaseModel):
        vacancy_str = vacancy_json.model_dump_json()
    elif isinstance(vacancy_json, dict):
        vacancy_str = json.dumps(vacancy_json, ensure_ascii=False)
    else:
        vacancy_str = str(vacancy_json)

    # Load config to get the preferred scorer model
    cfg = get_config()
    model_name = cfg.get("pipeline", {}).get("scorer", {}).get("model", "google/gemini-2.5-flash")
    temperature = cfg.get("pipeline", {}).get("scorer", {}).get("temperature", 0.0)

    try:
        _, validated_dict = await run_agent(
            prompt=load_prompt("prompts/scorer.xml"),
            schema=RelevanceResponse,
            model=model_name,
            temperature=temperature,
            vacancy=vacancy_str,
            cv_text=cv_text
        )
        score = validated_dict.get("score", 0.5)
        return float(score)
    except SchemaValidationError as e:
        print(f"\t\t\t ⚠️  Scorer schema validation failed: {e}. Returning default score 0.5.")
        log_debug(f"[SCORER VALIDATION ERROR] {e}")
        return 0.5
    except Exception as e:
        print(f"Error in calculate_relevance: {e}")
        return 0.5
