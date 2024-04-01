import os
from datetime import datetime
from typing import Any

import google.generativeai as palm
from google.api_core.exceptions import ServiceUnavailable, InternalServerError, TooManyRequests, DeadlineExceeded

from src.llms.generative_model import GenerativeModel, GenerativeModelResponse


class PaLMGenerativeModel(GenerativeModel):
    model: str

    def __init__(self, **data: Any):
        palm.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        super().__init__(**data)

    def generate(self, prompt: str) -> GenerativeModelResponse:
        try:
            chat_completion = palm.chat(messages=[prompt])
            return GenerativeModelResponse(
                generated_text=chat_completion.last,
                prompt_token=palm.count_message_tokens(prompt=prompt)['token_count'],
                generated_token=palm.count_message_tokens(prompt=chat_completion.last)['token_count'],
                created_at=datetime.now()
            )
        except (ServiceUnavailable, InternalServerError, TooManyRequests, DeadlineExceeded) as e:
            print(f"Google (PaLM) API error: {e}")
            return self.generate(prompt)
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise e
