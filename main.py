import datetime
import json
from pathlib import Path
from typing import Annotated

import torch
import typer
from dotenv import load_dotenv
from llmlingua import PromptCompressor
from loguru import logger

from src.models.generative_model_response import GenerativeModelResponse
from src.prompts import story_data_generation_prompt, last_chapter_generation_prompt, ending_evaluation_prompt
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
        story = model.generate(prompt, temperature=None)
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
                                         f.name.startswith(f'{ending_type.name}_{story_data_file.name}_{approach}')])
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
                story = model.generate(prompt, temperature=None)
                logger.info(f"Story generated: {story}")

                output_json = json.loads(story.model_dump_json())

                with open(output_folder / f"{ending_type.name}_{story_data_file.name}_{approach}_influence_{i}.json",
                          'w') as f:
                    json.dump(output_json, f, indent=4)
                logger.info(
                    f"Story saved to: {output_folder / f'{ending_type.name}_{story_data_file.name}_{approach}_influence_{i}.json'}")
                i += 1


@app.command()
def evaluate_story(eval_model: Annotated[str, typer.Option()]):
    evaluation_model = get_generative_model(eval_model)
    logger.info(f"Using model: {eval_model}")

    root_outputs_path = Path("outputs")
    models = [f for f in root_outputs_path.glob(f"*") if f.is_dir() and f.name != "logs"]

    for generation_model in models:
        logger.info(f"Evaluating stories for model: {generation_model.name}")
        story_influence_folders = [f for f in generation_model.glob(f"story_influence/*") if f.is_dir()]

        for ending_type in story_influence_folders:
            logger.info(f"Evaluating stories in folder: {ending_type.name}")
            story_influence_files = [f for f in ending_type.glob("*.json")]

            output_path = root_outputs_path / generation_model.name / "evaluation" / ending_type.name
            output_path.mkdir(exist_ok=True, parents=True)

            i = 0
            for story_influence_file in story_influence_files:
                logger.info(f"Evaluating story: {story_influence_file}, {i + 1}/{len(story_influence_files)}")
                story_influence = GenerativeModelResponse.model_validate_json(story_influence_file.read_text())

                last_chapter = story_influence.generated_text
                logger.info(f"Last chapter: {last_chapter}")

                prompt = ending_evaluation_prompt.format(last_chapter_story=last_chapter)
                logger.info(f"Prompt: {prompt}")

                response = evaluation_model.generate(prompt, temperature=0.0)
                logger.info(f"Response: {response}")

                parsed_response = parse_json_string(response.generated_text)
                logger.info(f"Parsed response: {parsed_response}")

                response_obj = response.model_dump()
                approach = story_influence_file.name.split(".json_")[-1].split("_")[0]
                response_obj['approach'] = approach
                response_obj['generation_model'] = generation_model.name
                response_obj['evaluation_model'] = eval_model
                response_obj['from_story_data_ending_type'] = ending_type.name
                response_obj['generated_story_ending_type'] = parsed_response['ending']
                response_obj['created_at'] = datetime.datetime.strftime(response.created_at, "%Y-%m-%dT%H:%M:%S.%f")
                logger.info(f"Response object: {response_obj}")

                output_file_path = output_path / story_influence_file.name
                with open(output_file_path, 'w') as f:
                    json.dump(response_obj, f, indent=4)
                logger.info(f"Story influence saved to: {story_influence_file}")
                i += 1


@app.command()
def analyse():
    results = {}
    output_path = Path("outputs")
    generation_model_folders = [f for f in output_path.glob("*") if f.is_dir() and f.name != "logs"]
    for generation_model in generation_model_folders:
        evaluation_files = [f for f in generation_model.glob("evaluation/**/*.json")]
        logger.info(f"Found {len(evaluation_files)} evaluation files.")

        logger.info("Analysing evaluation files")
        for evaluation_file in evaluation_files:
            logger.info(f"Analysing evaluation file: {evaluation_file.name}")
            evaluation = json.loads(evaluation_file.read_text())
            generation_model = evaluation['generation_model']
            from_story_data_ending_type = evaluation['from_story_data_ending_type']
            generated_story_ending_type = evaluation['generated_story_ending_type']
            approach = evaluation['approach']

            if approach not in results:
                results[approach] = {}
            if generation_model not in results[approach]:
                results[approach][generation_model] = {}
            if from_story_data_ending_type not in results[approach][generation_model]:
                results[approach][generation_model][from_story_data_ending_type] = {}
                for ending_type in ['positive', 'negative', 'neutral']:
                    results[approach][generation_model][from_story_data_ending_type][ending_type] = 0
            if generated_story_ending_type not in results[approach][generation_model][from_story_data_ending_type]:
                results[approach][generation_model][from_story_data_ending_type][generated_story_ending_type] = 0

            results[approach][generation_model][from_story_data_ending_type][generated_story_ending_type] += 1

    with open(output_path / "analysis.json", 'w') as f:
        json.dump(results, f, indent=4)
        logger.info(f"Analysis saved to: {output_path / 'analysis.json'}")


if __name__ == '__main__':
    load_dotenv()
    Path("outputs/logs").mkdir(exist_ok=True, parents=True)
    logger.add("outputs/logs/{time}.log")
    app()
