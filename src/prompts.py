story_data_generation_prompt = """Please write a brief 300-word game story synopsis with a {ending_type} ending. 

Output format:
Title: title of the story
Story: story synopsis until ending

Output:
"""

last_chapter_generation_prompt = """Please write the last chapter of the given story given the following story title and synopsis.

Story Synopsis:
{story_synopsis}

Output format:
Title: title of the story
Last Chapter Story: last chapter of the story

Output:
"""

ending_evaluation_prompt = """Please identify the type of ending in this story. Please make sure to format your output as a code block using triple backticks (```json and ```).

{last_chapter_story}

Output format:
```json
{{
    "ending": "positive", "negative", or "neutral"
}}
```"""
