from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel

from src.models.generative_model_response import GenerativeModelResponse


class GenerativeModel(ABC, BaseModel):
    @abstractmethod
    def generate(self, prompt: str, temperature: Optional[float]) -> GenerativeModelResponse:
        pass
