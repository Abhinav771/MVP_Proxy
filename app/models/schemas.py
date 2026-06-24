from pydantic import BaseModel, field_validator

class ChatRequest(BaseModel):
    prompt: str

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v

class SetLimitRequest(BaseModel):
    ip: str
    limit: int