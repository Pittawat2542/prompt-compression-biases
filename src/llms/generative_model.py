from abc import ABC, abstractmethod

from pydantic import BaseModel

from src.models.generative_model_response import GenerativeModelResponse


class GenerativeModel(ABC, BaseModel):
    @abstractmethod
    def generate(self, prompt: str) -> GenerativeModelResponse:
        pass
