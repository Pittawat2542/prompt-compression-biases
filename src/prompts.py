story_data_generation_prompt = """Please write a brief 300-word game story synopsis with a {ending_type} ending. Please make sure to format your output as a code block using triple backticks (```json and ```)
Output format:
```json
{{
    "title": game title,
    "story": game story synopsis until ending
}}
```"""

last_chapter_generation_prompt = """Please write the last chapter of the given story given the following story title and synopsis. Please make sure to format your output as a code block using triple backticks (```json and ```).

Story title:
{story_title}

Story Synopsis:
{story_synopsis}

Output format:
```json
{{
    "title": game title,
    "last_chapter": story of last chapter
}}
```"""

ending_evaluation_prompt = """Please identify the type of ending in this story. Please make sure to format your output as a code block using triple backticks (```json and ```)
Title: {story_title}

Last Chapter Story:
{last_chapter_story}

Output format:
```json
{{
    "ending": "positive", "negative", or "neutral"
}}
```"""