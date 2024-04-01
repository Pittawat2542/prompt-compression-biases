from src.llms.anthropic_gen_model import AnthropicGenerativeModel
from src.llms.generative_model import GenerativeModel
from src.llms.google_gen_model import GoogleGenerativeModel
from src.llms.openai_gen_model import OpenAIGenerativeModel
from src.llms.palm_model import PaLMGenerativeModel


def get_generative_model(model: str) -> GenerativeModel:
    if model in ['gemini-1.0-pro']:
        return GoogleGenerativeModel(model=model)
    elif model in ['chat-bison-001']:
        return PaLMGenerativeModel(model=model)
    elif model in ['gpt-3.5-turbo-0125', 'gpt-4-0125-preview']:
        return OpenAIGenerativeModel(model=model)
    elif model in ['claude-2.1', 'claude-3-opus-20240229']:
        return AnthropicGenerativeModel(model=model)
    else:
        raise ValueError(f"Model {model} not supported")
