from pydantic import BaseModel, Field

class DraftCV(BaseModel):
    content: str = Field(description="The content of the tailored resume in Markdown format")

class RefinementResult(BaseModel):
    data: DraftCV = Field(description="The refined draft CV")
    improvement_notes: str = Field(description="Notes on improvements made")
    alignment_score: float = Field(description="A score from 0 to 100 on how well it fits the vacancy without lying")
