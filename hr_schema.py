from pydantic import BaseModel, Field

class HRScore(BaseModel):
    score: int = Field(description="Score from 0 to 10 evaluating how well the resume matches the vacancy.")
    reasoning: str = Field(description="Brief explanation of the score.")
