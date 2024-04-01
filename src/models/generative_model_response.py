import datetime

from pydantic import BaseModel


class GenerativeModelResponse(BaseModel):
    generated_text: str
    prompt_token: int
    generated_token: int
    created_at: datetime.datetime
