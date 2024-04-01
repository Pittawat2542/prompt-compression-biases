from datetime import datetime
from typing import Any

from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError, APIError

from src.llms.generative_model import GenerativeModel, GenerativeModelResponse


class OpenAIGenerativeModel(GenerativeModel):
    model: str

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._client = OpenAI()

    def generate(self, prompt: str) -> GenerativeModelResponse:
        try:
            chat_completion = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return GenerativeModelResponse(
                generated_text=chat_completion.choices[0].message.content,
                prompt_token=chat_completion.usage.prompt_tokens,
                generated_token=chat_completion.usage.completion_tokens,
                created_at=datetime.fromtimestamp(chat_completion.created)
            )
        except (APITimeoutError, APIConnectionError, RateLimitError, APIError) as e:
            print(f"OpenAI API error: {e}")
            return self.generate(prompt)
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise e
