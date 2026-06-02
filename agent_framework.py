import os
import time
from google import genai
from google.genai.errors import ServerError, ClientError
from pydantic import BaseModel
from typing import Type, TypeVar, Dict, Any

T = TypeVar("T", bound=BaseModel)

from dotenv import load_dotenv
load_dotenv()

class Agent:
    def __init__(self, prompt_path: str, model_name: str = 'gemini-flash-lite-latest'):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model_name = model_name
        self.prompt_path = prompt_path
        with open(self.prompt_path, 'r') as f:
            self.prompt_template = f.read()

    def run(self, schema: Type[T], params: Dict[str, Any]) -> T:
        prompt = self.prompt_template.format(**params)
        
        retries = 5
        backoff = 2
        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.0
                    )
                )
                return schema.model_validate_json(response.text)
            except (ServerError, ClientError) as e:
                if isinstance(e, ClientError) and e.status_code != 429:
                    raise e
                if attempt == retries - 1:
                    raise e
                
                print(f"Gemini API Error (retriable: {e.status_code if hasattr(e, 'status_code') else '503'}). Retrying in {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2
