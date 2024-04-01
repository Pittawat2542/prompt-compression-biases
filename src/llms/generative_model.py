import datetime
from abc import ABC, abstractmethod

from pydantic import BaseModel


class GenerativeModelResponse(BaseModel):
    generated_text: str
    prompt_token: int
    generated_token: int
    created_at: datetime.datetime


class GenerativeModel(ABC, BaseModel):
    @abstractmethod
    def generate(self, prompt: str) -> GenerativeModelResponse:
        pass
