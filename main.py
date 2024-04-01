import json
from pathlib import Path
from typing import Annotated

import torch
import typer
from dotenv import load_dotenv
from llmlingua import PromptCompressor
from loguru import logger

from src.models.generative_model_response import GenerativeModelResponse
from src.prompts import story_data_generation_prompt, last_chapter_generation_prompt
from utils.llms import get_generative_model

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

        output_json = json.loads(story.model_dump_json())

        with open(output_folder / f"story_{i}.json", 'w') as f:
            json.dump(output_json, f, indent=4)
        logger.info(f"Story saved to: {output_folder / f'story_{i}.json'}")
        i += 1


@app.command()
def generate_story_influence(gen_model: Annotated[str, typer.Option()], approach: Annotated[str, typer.Option()],
                             n_per_story_data: Annotated[int, typer.Option] = 1):
    if approach not in ['baseline', 'compressed']:
        typer.echo("Approach must be either 'baseline' or 'compressed'")
        raise typer.Exit(code=1)

    all_story_data_path = Path(f"outputs/claude-3-opus-20240229/story_data")
    story_data_folders = [f for f in all_story_data_path.iterdir() if f.is_dir()]
    logger.info(f"Story data folders: {story_data_folders}")

    model = get_generative_model(gen_model)
    logger.info(f"Using model: {gen_model}")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    llm_lingua = None
    if approach == 'compressed':
        llm_lingua = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            device_map=device
        )
        logger.info(
            f"Using llmlingua for prompt compression using model: 'microsoft/llmlingua-2-xlm-roberta-large-meetingbank'")

    for ending_type in story_data_folders:
        logger.info(f"Generating stories for ending type: {ending_type}")
        story_data_files = [f for f in ending_type.glob("*.json")]

        for story_data_file in story_data_files:
            logger.info(f"Generating stories for story data: {story_data_file}")

            output_folder = Path(f"outputs/{gen_model}/story_influence/{ending_type.name}")
            output_folder.mkdir(exist_ok=True, parents=True)
            logger.info(f"Saving generated stories to: {output_folder}")

            num_generated_stories = len([f for f in output_folder.glob("*.json") if
                                         f.name.startswith(f'{ending_type.name}_{story_data_file.name}')])
            logger.info(f"Already generated {num_generated_stories} stories")

            story_data = GenerativeModelResponse.model_validate_json(story_data_file.read_text())

            story_synopsis = story_data.generated_text
            if approach == 'compressed':
                story_synopsis = llm_lingua.compress_prompt([story_synopsis], rate=0.5, force_tokens=['\n', '?'])[
                    'compressed_prompt']

            prompt = last_chapter_generation_prompt.format(story_synopsis=story_synopsis)

            i = num_generated_stories
            while i < n_per_story_data:
                logger.info(f"Generating story {i + 1}/{n_per_story_data}")
                logger.info(f"Start generating story with story data: {story_data}")
                story = model.generate(prompt)
                logger.info(f"Story generated: {story}")

                output_json = json.loads(story.model_dump_json())

                with open(output_folder / f"{ending_type.name}_{story_data_file.name}_influence_{i}.json", 'w') as f:
                    json.dump(output_json, f, indent=4)
                logger.info(
                    f"Story saved to: {output_folder / f'{ending_type.name}_{story_data_file.name}_influence_{i}.json'}")
                i += 1


@app.command()
def evaluate_story():
    pass


if __name__ == '__main__':
    load_dotenv()
    Path("outputs/logs").mkdir(exist_ok=True, parents=True)
    logger.add("outputs/logs/{time}.log")
    app()
