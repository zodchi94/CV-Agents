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
    """Load a prompt template from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# Alias kept for backward compatibility with main.py imports
load_file = load_prompt


def clean_json_string(text: str | None) -> str:
    """Strip Markdown code fences and whitespace from a JSON string."""
    if not text:
        return ""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
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


def _is_truncated_response(text: str | None) -> bool:
    """Detect if the model response was truncated mid-generation.
    A truncated JSON response typically ends without a closing brace/bracket,
    or ends with an unterminated string value.
    Returns False for None or empty input.
    """
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    # Count unescaped braces to detect unclosed JSON objects
    depth = 0
    in_string = False
    escape_next = False
    for char in stripped:
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
    # If depth > 0, the JSON object was never closed — response was truncated
    return depth > 0 or in_string


def safe_json_loads(text: str) -> dict:
    """Parse JSON from model response, handling common issues:
    - Markdown code block wrappers (```json ... ```)
    - Extra text/data after the JSON object
    - Literal control characters inside string values
    - Truncated responses (raises JSONDecodeError so the candidate is skipped)
    """
    text = clean_json_string(text)

    # Detect truncated responses before attempting to parse
    if _is_truncated_response(text):
        raise json.JSONDecodeError("Response appears to be truncated (unclosed JSON object)", text, len(text))

    # First attempt: standard parse
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Second attempt: extract the first complete JSON object via raw_decode
        # This handles "Extra data" errors where the model appends text after JSON
        if "Extra data" in str(e):
            try:
                obj, _ = json.JSONDecoder().raw_decode(text)
                return obj
            except json.JSONDecodeError:
                pass
        # Third attempt: escape control characters and retry
        fixed = _escape_control_chars_in_json(text)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            obj, _ = json.JSONDecoder().raw_decode(fixed)
            return obj


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


async def run_agent(prompt: str, schema, model: str, temperature: float, max_retries: int = 2, **kwargs) -> tuple[str, dict]:
    # Formats keyword arguments safely
    # If a value in kwargs is not a string, let's convert it to string or JSON
    formatted_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, (dict, list, BaseModel)):
            if v is not None and hasattr(v, "model_dump_json"):
                formatted_kwargs[k] = v.model_dump_json()
            elif v is not None:
                formatted_kwargs[k] = json.dumps(v, ensure_ascii=False)
            else:
                formatted_kwargs[k] = "None"  # Explicitly set to "None" or an empty string if v is None
        elif v is not None:
            formatted_kwargs[k] = str(v)
        else:
            formatted_kwargs[k] = "None"  # Explicitly set to "None" if v is None
    
    try:
        formatted_prompt = prompt.format(**formatted_kwargs)
    except KeyError as e:
        # Handle cases where prompt expects a key not present in formatted_kwargs
        # This might happen if kwargs are conditionally passed
        print(f"Warning: Placeholder {e} not found in prompt. Skipping formatting for this placeholder.")
        # For now, we'll just return the original prompt if a key is missing
        formatted_prompt = prompt
    except Exception as e:
        print(f"An unexpected error occurred during prompt formatting: {e}")
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
    
    last_exception = None
    for attempt in range(max_retries + 1):
        async with httpx.AsyncClient(timeout=300.0) as client:
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

            try:
                parsed = safe_json_loads(content)
            except json.JSONDecodeError as e:
                last_exception = e
                if attempt < max_retries:
                    log_debug(
                        f"[RETRY {attempt+1}/{max_retries}] JSON parse error for model {model}: {e}\n"
                        f"Raw content (first 500 chars): {content[:500]}\n"
                    )
                    print(f"\n\t\t\t ⚠️  JSON parse error (attempt {attempt+1}/{max_retries+1}), retrying...")
                    await asyncio.sleep(1)
                    continue
                else:
                    log_debug(
                        f"[FINAL FAILURE] JSON parse error for model {model} after {max_retries+1} attempts: {e}\n"
                        f"Raw content (first 500 chars): {content[:500]}\n"
                    )
                    raise
            
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
    
    # Should not reach here, but just in case
    raise last_exception or RuntimeError(f"run_agent failed after {max_retries+1} attempts for model {model}")


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
