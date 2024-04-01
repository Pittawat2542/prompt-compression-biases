import json
import re


def parse_json_string(json_string: str) -> dict:
    if json_string.startswith("{") and json_string.endswith("}"):
        return json.loads(json_string)

    pattern = r'```(json)?\n([\s\S]*?)(?<!`)```'
    match = re.findall(pattern, json_string, re.DOTALL)

    if match is None or len(match) == 0:
        raise ValueError(
            "JSON markdown block not found in the message. Please use the following format:\n```json\n{...}\n```")

    json_string = match[-1][-1].strip()
    json_string = json.dumps(json_string)
    return json.loads(json_string)
