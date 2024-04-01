import json
from json import JSONDecodeError
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from loguru import logger

from src.prompts import story_data_generation_prompt
from src.models.generative_model_response import GenerativeModelResponse
from utils.llms import get_generative_model
from utils.parsers import parse_json_string

app = typer.Typer()


@app.command()
def generate_story_data(gen_model: Annotated[str, typer.Option()], ending_type: Annotated[str, typer.Option()],
                        n: Annotated[int, typer.Option] = 1):
    if ending_type not in ['positive', 'negative', 'neutral']:
        typer.echo("Ending type must be either 'positive', 'negative', or 'neutral'")
        raise typer.Exit(code=1)

    if n < 1:
        typer.echo("Number of stories to generate must be at least 1")
        raise typer.Exit(code=1)

    model = get_generative_model(gen_model)
    prompt = story_data_generation_prompt.format(ending_type=ending_type)

    logger.info(f"Generating {n} stories with ending type: {ending_type} using model: {gen_model}")

    output_folder = Path(f"outputs/{gen_model}/story_data/{ending_type}")
    output_folder.mkdir(exist_ok=True, parents=True)
    logger.info(f"Saving generated stories to: {output_folder}")

    n_generated_stories = len(list(output_folder.glob("*.json")))
    logger.info(f"Already generated {n_generated_stories} stories")

    i = n_generated_stories
    while i < n:
        logger.info(f"Generating story {i + 1}/{n}")
        logger.info(f"Start generating story with prompt: {prompt}")
        story = model.generate(prompt)
        logger.info(f"Story generated: {story}")
        try:
            parsed_content = parse_json_string(story.generated_text)
        except JSONDecodeError as e:
            logger.error(f"Error generating story: {e}")
            continue
        logger.info(f"Story parsed: {parsed_content}")

        output_json = json.loads(story.model_dump_json())
        output_json['parsed'] = parsed_content

        with open(output_folder / f"story_{i}.json", 'w') as f:
            json.dump(output_json, f, indent=4)
        logger.info(f"Story saved to: {output_folder / f'story_{i}.json'}")
        i += 1


@app.command()
def generate_story_influence():
    pass


@app.command()
def evaluate_story():
    pass


if __name__ == '__main__':
    load_dotenv()
    Path("outputs/logs").mkdir(exist_ok=True, parents=True)
    logger.add("outputs/logs/{time}.log")
    app()
