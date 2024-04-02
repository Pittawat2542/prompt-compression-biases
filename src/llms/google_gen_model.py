import os
from datetime import datetime
from typing import Any, Optional

import google.generativeai as genai
from google.ai.generativelanguage_v1 import HarmCategory
from google.api_core.exceptions import ServiceUnavailable, InternalServerError, TooManyRequests, DeadlineExceeded
from google.generativeai import GenerationConfig
from google.generativeai.types import HarmBlockThreshold

from src.llms.generative_model import GenerativeModel
from src.models.generative_model_response import GenerativeModelResponse

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
}


class GoogleGenerativeModel(GenerativeModel):
    model: str

    def __init__(self, **data: Any):
        super().__init__(**data)
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self._client = genai.GenerativeModel(self.model)

    def generate(self, prompt: str, temperature: Optional[float]) -> GenerativeModelResponse:
        try:
            chat = self._client.start_chat()
            chat_completion = chat.send_message(prompt, safety_settings=safety_settings,
                                                generation_config=GenerationConfig(temperature=temperature))
            return GenerativeModelResponse(
                generated_text=chat_completion.text,
                prompt_token=self._client.count_tokens(prompt).total_tokens,
                generated_token=self._client.count_tokens(chat_completion.text).total_tokens,
                created_at=datetime.now()
            )
        except (ServiceUnavailable, InternalServerError, TooManyRequests, DeadlineExceeded) as e:
            print(f"Google API error: {e}")
            return self.generate(prompt, temperature)
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise e
