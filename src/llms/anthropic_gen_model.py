from datetime import datetime
from typing import Any, Optional

from anthropic import Anthropic, APITimeoutError, APIConnectionError, RateLimitError, APIError

from src.llms.generative_model import GenerativeModel
from src.models.generative_model_response import GenerativeModelResponse


class AnthropicGenerativeModel(GenerativeModel):
    model: str

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._client = Anthropic()

    def generate(self, prompt: str, temperature: Optional[float]) -> GenerativeModelResponse:
        try:
            chat_completion = self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return GenerativeModelResponse(
                generated_text=chat_completion.content[0].text,
                prompt_token=chat_completion.usage.input_tokens,
                generated_token=chat_completion.usage.output_tokens,
                created_at=datetime.now()
            )
        except (APITimeoutError, APIConnectionError, RateLimitError, APIError) as e:
            print(f"Anthropic API error: {e}")
            return self.generate(prompt)
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise e
