"""
Open Research Knowledge Graph (ORKG) Properties Extractor: Automated ORKG Properties Extraction and Evaluation Using
GPT-3.5-Turbo.
This script processes research problem descriptions of scientific publications to extract properties (also called
dimensions to distinguish from ORKG properties) using the Large Language Model (LLM) GPT-3.5-Turbo.

It performs extraction, evaluation, and matching of LLM extracted properties to ORKG properties that are fetched from
the ORKG API by the program.

The script follows these main steps:
1. Data Preparation: Reads gold standard and comparison data from a CSV file and structures it for processing.
2. Properties Extraction: Extracts properties (also called dimensions to distinguish from ORKG properties) from publications using different prompting strategies with the LLM
GPT-3.5-Turbo.
3. Evaluation: Assesses the extracted properties based on alignment, deviation and the number of mappings with ORKG
properties (gold standard).
4. Matching: Matches the extracted properties to ORKG properties URIs retrieved via the API.
5. Output: Prints and optionally saves evaluation results to txt and svg files, and writes all data to JSON.

Classes:
    DualOutput: Handles simultaneous output to both the console and a file.

Functions:
    read_csv(file_name) -> dict: Reads a CSV file and converts the content to a dictionary.
    extract_dimensions_one(publication, llm) -> None: Extracts dimensions from a single publication.
    query_gpt_agent(prompt, messages_history, llm) -> str: Queries the selected LLM API with a given prompt.
    load_prompts_from_yaml(file_name: str) -> List[str]: Loads the prompts from yaml files and returns them as a list.
    eval_dimensions_one(publication, llm) -> None: Evaluates dimensions for a single publication.
    eval_dimensions_all(publications, llm) -> None: Evaluates dimensions for a list of publications.
    postprocessing_response(string) -> list: Cleans and extracts a Python list from a raw LLM response.
    store_csv_data(table) -> list: Converts relevant CSV table data to a list of dictionaries.
    dimensions_and_eval_to_json(publications) -> None: Saves the publications dict including results and evaluations to
    a JSON file.
    match_dimension_to_orkg(dimension, api_orkg_properties) -> dict or None: Maps a dimension to ORKG properties.
    save_api_orkg_properties_pickle() -> None: Fetches ORKG properties from the API and saves them to a pickle file.
    fetch_all_properties(url) -> list: Retrieves all properties from the ORKG API.
    preprocessing_dimensions(dimensions_to_map) -> None: Preprocesses dimensions for matching to ORKG properties.
    match_dimensions_to_orkg(publication) -> None: Matches extracted dimensions to existing ORKG properties.
    eval_match_dimensions_to_orkg(publications) -> None: Evaluates matches to existing ORKG properties.
    print_eval_details(publications) -> None: Prints evaluation details to the console or saves them to a file.
    export_svg_bar_chart(label, min, max, strat_strings, chart_values, file_name) -> None: Exports a horizontal bar
    chart to an SVG file.
    export_svg_heatmap(strat_strings, heatmap_values, file_name) -> None: Exports a heatmap of sign test results to an
    SVG file.
    calc_sign_test(publications) -> dict: Calculates the sign test of the differences of alignment, deviation and
    number of mappings between two selected prompting strategies.
    to_int(string, replacement, index1, index2) -> int: Converts the string of the LLM response to int. If converting
    fails, replacement values are returned.

Usage:
    Before running the script, install dependencies with:
    $ pip install -r requirements.txt

    Required files:
    - requirements.txt: Contains dependencies that have to be installed before running the script
    - .env: Stores OpenAI API key and organization ID.
    - dimensions_system_prompts.yaml: Contains prompts for dimensions extraction.
    - dimensions_eval_system_prompts.yaml: Contains prompts for evaluation of dimensions extraction.
    - orkg_properties_llm_dimensions_dataset_test.csv: Contains the dataset created by Nechakhin et al., 2024 (slightly
    modified as three rows are removed in order to use them as few-shot examples in one of the prompts.

    Run the script:
    $ python orkg_properties_extraction.py
"""
import math
import pandas as pd
import json
import sys
import requests
import os
import pickle
import yaml
from typing import List, Dict, Union
from dotenv import load_dotenv
from openai import OpenAI
import inflect
from scipy.stats import binomtest
import matplotlib.pyplot as plt
import numpy as np


class DualOutput:
    """
    A class to handle simultaneous output to both the console and a file.

    Attributes:
        file (TextIO): A file object to write the output to.
        console (object): The original standard output (sys.stdout).

    Methods:
        write(message: str):
            Writes a message to both the console and the file.

        flush():
            Flushes both the console and the file buffers.
    """

    def __init__(self, file):
        """
        Initializes the DualOutput class with a file object.

        Args:
            file (TextIO): A file object where the output will be written.
        """
        self.file = file
        self.console = sys.stdout

    def write(self, message):
        """
        Writes a message to both the console and the file.

        Args:
            message (str): The message to write.
        """
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        """
        Flushes both the console and the file buffers.
        """
        self.console.flush()
        self.file.flush()


def read_csv(file_name: str) -> dict:
    """
    Reads a CSV file and converts its content into a dictionary.

    Args:
        file_name (str): The name of the CSV file.

    Returns:
        dict: A dictionary where each column is represented as a key, with its values as lists.
    """
    df = pd.read_csv(file_name, sep=";")
    table = {column: df[column].tolist() for column in df.columns}
    return table


def extract_dimensions_one(publication: dict, llm: int = 0) -> None:
    """
    Extracts dimensions from a single publication using different prompting strategies with LLM APIs.

    Args:
        publication (dict): A dictionary containing publication details including the research problem.
        llm (int, optional): Specifies the LLM type (0 for GPT, 1 for Gemini). Defaults to 0.

    Returns:
        None
    """
    dimensions_system_prompts = load_prompts_from_yaml("dimensions_system_prompts.yaml")
    research_problem = publication["research_problem"]
    dimensions = list()
    for i in range(len(dimensions_system_prompts)):
        messages_history = list()
        messages_history.append({"role": "system", "content": dimensions_system_prompts[i]})
        print("Sending prompt to llm agent...")  # In the terminal the user can see the progress of the extraction
        # and evaluation process
        response = query_gpt_agent(research_problem, messages_history, llm)
        print("Received response by llm agent.")
        liste = postprocessing_response(response)
        dimensions.append(liste)
    publication["dimensions"] = dimensions


def query_gpt_agent(prompt: str, messages_history: List[Dict[str, str]], llm: int = 0) -> str:
    """
    Sends a prompt to the selected LLM API and retrieves the response.

    Args:
        prompt (str): The query to be sent to the LLM.
        messages_history (List[Dict[str, str]]): The conversation history to include in the API request.
        llm (int, optional): Specifies the LLM type (0 for GPT, 1 for Gemini). Defaults to 0.

    Returns:
        str: The response from the LLM.
    """
    api_key = os.getenv("API_KEY_AGENT")
    if not api_key:
        raise ValueError(
            "API key for OpenAI access not found. Please set the API_KEY_AGENT environment variable.")

    organization = os.getenv("ORGANIZATION")  # the ORGANIZATION ID must be stored in the .env file.
    if not organization:
        raise ValueError(
            "Organization for OpenAI access not found. Please set the ORGANIZATION environment variable.")

    client = OpenAI(
        organization=organization,
        api_key=api_key
    )
    if llm == 0:
        model = "gpt-3.5-turbo-0125"
    else:
        model = "gpt-3.5-turbo-1106"
    messages_history.append({"role": "user", "content": prompt})
    # messages format according to the OpenAI API reference: https://platform.openai.com/docs/api-reference/
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages_history,
            temperature=0,  # temperature was set to 0 according to OpenAI guidelines for minimizing non-determinism
            # (https://platform.openai.com/docs/guides/text-generation/reproducible-outputs and
            # https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api)
            seed=4  # use of seed parameter to promote reproducible outputs
            # (https://platform.openai.com/docs/guides/text-generation/reproducible-outputs)
        )
        response = response.choices[0].message.content.strip()
        return response

    except Exception as e:
        print(f"Error querying OpenAI API: {e}")
        return ""


def load_prompts_from_yaml(file_name: str) -> List[str]:
    """
    Load prompts from a YAML file.

    Args:
        file_name (str): The name of the YAML file containing the prompts.

    Returns:
        prompts (List [str]): A list of prompts.
    """
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    prompts = list()
    try:
        with open(file_path, 'r') as file:
            prompts_yaml = yaml.safe_load(file)
            prompts = [prompt.strip() for prompt in prompts_yaml]
        return prompts

    except FileNotFoundError as fnf_error:
        print(f"Error: The file '{file_name}' was not found. {fnf_error}")
        return prompts

    except yaml.YAMLError as yaml_error:
        print(f"Error: Failed to parse YAML file '{file_name}'. {yaml_error}")
        return prompts

    except Exception as e:
        print(f"An unexpected error while handling the yaml file {file_name} occurred: {e}")
        return prompts


def eval_dimensions_one(publication: dict, llm: int) -> None:
    """
    Evaluates the dimensions extracted from a single publication based on multiple criteria.

    Args:
        publication (dict): A dictionary containing the publication details and dimensions.
        llm (int): Specifies the LLM type used for evaluation.

    Returns:
        None
    """
    dimensions_eval_system_prompts = load_prompts_from_yaml("dimensions_eval_system_prompts.yaml")
    # each of the 5 dimensions lists (format string) will be evaluated:
    five = list()
    for liste in publication["dimensions"]:
        if isinstance(liste, list):
            liste = " ".join(map(str, liste))
        if isinstance(publication["orkg_properties"], list):
            publication["orkg_properties"] = " ".join(map(str, publication["orkg_properties"]))
        prompt = liste + "\n" + publication["orkg_properties"]
        # three prompts for each list:
        three = list()
        for i in range(len(dimensions_eval_system_prompts)):
            messages_history = list()
            messages_history.append({"role": "system", "content": dimensions_eval_system_prompts[i]})
            print("Sending prompt to llm agent...")  # In the terminal the user can see the progress of the extraction
            # and evaluation process
            response = query_gpt_agent(prompt, messages_history, llm)
            print("Received response by llm agent.")
            three.append(response)
        five.append(three)
    publication["eval"] = five


def eval_dimensions_all(publications: List[dict], llm: int) -> None:
    """
    Evaluates dimensions for a list of publications.

    Args:
        publications (list): A list of dictionaries containing publication details and dimensions.
        llm (int): Specifies the LLM type used for evaluation.

    Returns:
        None
    """
    for publication in publications:
        eval_dimensions_one(publication, llm)


def print_eval_details(publications: List[dict]) -> None:
    """
    Prints evaluation details for all publications and optionally saves them to a text file.
    Exports svg files for 3 bar charts and one heatmap with the evaluation results.

    Args:
        publications (list): A list of dictionaries containing evaluation details.

    Returns:
        None
    """
    print("Would you like to save the evaluation details to a txt-file and export svg-charts?")
    print("A: Yes")
    print("B: No")
    svg_name = ""
    while True:
        choice = input("Please enter A or B: ")
        if choice.upper() == "A":
            file_bool = True
            file_name = input("Please enter a file name to store the evaluation details "
                              "(without the endings .txt or .svg): ")
            svg_name = file_name
            file_name = file_name + ".txt"
            break
        elif choice.upper() == "B":
            file_bool = False
            file_name = "file_name.txt"
            break
        else:
            print("Invalid input. Please enter 'A' or 'B'")

    with (open(file_name, 'w', encoding='utf-8', errors="ignore") as file):
        if file_bool:
            dual_output = DualOutput(file)
            sys.stdout = dual_output  # "print" function will output the given strings both to the console and to
            # the given file
            count = 1
            sums = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
            for publication in publications:
                sums[0][0] += int(publication["nechakhin_alignment"])
                sums[0][1] += int(publication["nechakhin_deviation"])
                sums[0][2] += int(publication["nechakhin_mappings"])
                for i in range(1, 6):
                    for j in range(3):

                        sums[i][j] += to_int(publication["eval"][i - 1][j], 3, i-1, j)
                        # On average once per 150 publications, a ValueError occurs because the
                        # LLM doesn't output an integer as it is supposed to do. Instead, it outputs something like
                        # "['approach', 'published_in:journal', 'issue', 'addresses']"
                        # During all the test runs, this was only the case for the alignment values, so it was
                        # decided to add +3 to the sum to not skew the results significantly (also in case this
                        # happens for the deviation int)
                        # Additionally, the error cases are printed so it can be decided to rerun the experiment if
                        # too many errors occur

                print("-" * 100)
                print(f"Publication {str(count)} of {str(len(publications))}:")
                count += 1
                print("Research problem: " + publication["research_problem"])
                print("-" * 100)
                print(
                    f"{'':<25} {'Alignment':<25} {'Deviation':<25}"
                    f" {'Mappings':<25}")  # table format with 4 columns of 25 characters length
                print("-" * 100)
                print(f'{"nechakhin_result":<25} {publication["nechakhin_alignment"]:<25} {publication["nechakhin_deviation"]:<25} {publication["nechakhin_mappings"]:<25}')
                print("-" * 100)
                print(f'{"zero-shot":<25} {publication["eval"][0][0]:<25} {publication["eval"][0][1]:<25} {publication["eval"][0][2]:<25}')
                print("-" * 100)
                print(
                    f'{"few-shot":<25} {publication["eval"][1][0]:<25} {publication["eval"][1][1]:<25} {publication["eval"][1][2]:<25}')
                print("-" * 100)
                print(
                    f'{"chain-of-thought":<25} {publication["eval"][2][0]:<25} {publication["eval"][2][1]:<25} {publication["eval"][2][2]:<25}')
                print("-" * 100)
                print(
                    f'{"combined":<25} {publication["eval"][3][0]:<25} {publication["eval"][3][1]:<25} {publication["eval"][3][2]:<25}')
                print("-" * 100)
                print(
                    f'{"optimized":<25} {publication["eval"][4][0]:<25} {publication["eval"][4][1]:<25} {publication["eval"][4][2]:<25}')
                print("-" * 100)
            print("-" * 100)
            n = len(publications)
            print(f"Overall mean of {str(n)} publications:")
            print("-" * 100)
            print(
                f"{'':<25} {'Alignment':<25} {'Deviation':<25}"
                f" {'Mappings':<25}")  # table format with 4 columns of 25 characters length
            print("-" * 100)
            print(
                f'{"nechakhin_result":<25} {str(round(sums[0][0]/float(n),2)):<25} {str(round(sums[0][1]/float(n),2)):<25} {str(round(sums[0][2]/float(n),2)):<25}')
            print("-" * 100)
            print(
                f'{"zero-shot":<25} {str(round(sums[1][0]/float(n),2)):<25} {str(round(sums[1][1]/float(n),2)):<25} {str(round(sums[1][2]/float(n),2)):<25}')
            print("-" * 100)
            print(
                f'{"few-shot":<25} {str(round(sums[2][0]/float(n),2)):<25} {str(round(sums[2][1]/float(n),2)):<25} {str(round(sums[2][2]/float(n),2)):<25}')
            print("-" * 100)
            print(
                f'{"chain-of-thought":<25} {str(round(sums[3][0]/float(n),2)):<25} {str(round(sums[3][1]/float(n),2)):<25} {str(round(sums[3][2]/float(n),2)):<25}')
            print("-" * 100)
            print(
                f'{"combined":<25} {str(round(sums[4][0]/float(n),2)):<25} {str(round(sums[4][1]/float(n),2)):<25} {str(round(sums[4][2]/float(n),2)):<25}')
            print("-" * 100)
            print(
                f'{"optimized":<25} {str(round(sums[5][0] / float(n), 2)):<25} {str(round(sums[5][1] / float(n), 2)):<25} {str(round(sums[5][2] / float(n), 2)):<25}')
            print("-" * 100)
            print("-" * 100)
            print("Significance: Sign test results comparing the prompting strategies: ")
            print("-" * 100)
            sign_test_results = [[None for _ in range(5)] for _ in range(5)]
            # Is few-shot better than zero-shot?
            sign_test_results[1][0] = calc_sign_test(publications, 0, 1)
            # Is chain-of-thought better than zero-shot?
            sign_test_results[2][0] = calc_sign_test(publications, 0, 2)
            # Is combined better than zero-shot?
            sign_test_results[3][0] = calc_sign_test(publications, 0, 3)
            # Is optimized better than zero-shot?
            sign_test_results[4][0] = calc_sign_test(publications, 0, 4)
            # Is combined better than few-shot?
            sign_test_results[3][1] = calc_sign_test(publications, 1, 3)
            # Is combined better than chain-of-thought?
            sign_test_results[3][2] = calc_sign_test(publications, 2, 3)
            # Is optimized better than few-shot?
            sign_test_results[4][1] = calc_sign_test(publications, 1, 4)
            # Is optimized better than chain-of-thought?
            sign_test_results[4][2] = calc_sign_test(publications, 2, 4)
            # Is optimized better than combined?
            sign_test_results[4][3] = calc_sign_test(publications, 3, 4)
            strat_strings = ["zero-shot", "few-shot", "chain-of-thought", "combined", "optimized"]
            print("p-values for alignment differences:")
            print("-" * 150)
            print(
                f"{'':<25}{'zero-shot':<25}{'few-shot':<25}{'chain-of-thought':<25}{'combined':<25}{'optimized':<25}")
            print("-" * 150)
            # table format with 6 columns of 25 characters length
            for i in range(5):
                alignment_values = [
                    f"{(f'{float(value):.3e}' if float(value) < 0.001 else f'{float(value):.3f}') if value is not None else '':<25}"
                    for j in range(5)
                    for value in [sign_test_results[i][j]['alignment'] if sign_test_results[i][j] is not None else None]
                ]
                print(f"{strat_strings[i]:<25}" + "".join(alignment_values))
                print("-" * 150)
            print("-" * 150)
            print("p-values for deviation differences:")
            print("-" * 150)
            print(
                f"{'':<25}{'zero-shot':<25}{'few-shot':<25}{'chain-of-thought':<25}{'combined':<25}{'optimized':<25}")
            print("-" * 150)
            # table format with 6 columns of 25 characters length
            for i in range(5):
                deviation_values = [
                    f"{(f'{float(value):.3e}' if float(value) < 0.001 else f'{float(value):.3f}') if value is not None else '':<25}"
                    for j in range(5)
                    for value in [sign_test_results[i][j]['deviation'] if sign_test_results[i][j] is not None else None]
                ]

                print(f"{strat_strings[i]:<25}" + "".join(deviation_values))
                print("-" * 150)
            print("-" * 150)
            print("p-values for mappings differences:")
            print("-" * 150)
            print(
                f"{'':<25}{'zero-shot':<25}{'few-shot':<25}{'chain-of-thought':<25}{'combined':<25}{'optimized':<25}")
            # table format with 6 columns of 25 characters length
            print("-" * 150)
            for i in range(5):
                mappings_values = [
                    f"{(f'{float(value):.3e}' if float(value) < 0.001 else f'{float(value):.3f}') if value is not None else '':<25}"
                    for j in range(5)
                    for value in [sign_test_results[i][j]['mappings'] if sign_test_results[i][j] is not None else None]
                ]

                print(f"{strat_strings[i]:<25}" + "".join(mappings_values))
                print("-" * 150)
            print("-" * 150)

            if file_bool:
                # Restore the original stdout
                sys.stdout = dual_output.console
    # svg files are only exported if file_bool is True
    if not file_bool:
        return
    # data preparation for heatmap export with significance values:
    sign_test_alignment = [
        [cell["alignment"] if cell is not None else None for cell in row]
        for row in sign_test_results
    ]

    sign_test_deviation = [
        [cell["deviation"] if cell is not None else None for cell in row]
        for row in sign_test_results
    ]

    sign_test_mappings = [
        [cell["mappings"] if cell is not None else None for cell in row]
        for row in sign_test_results
    ]

    # Row and column labels
    strat_strings = ["zero-shot", "few-shot", "chain-of-thought", "combined", "optimized"]

    # For each alignment, deviation, mappings:
    # 1 Create a matrix with values or empty strings
    # 2 Create heatmap out of matrix and export svg
    alignment_values = np.array([[cell if cell is not None else "" for cell in row] for row in sign_test_alignment])
    export_svg_heatmap(strat_strings, alignment_values, svg_name + "_heatmap_alignment.svg")
    deviation_values = np.array(
        [[cell if cell is not None else "" for cell in row] for row in sign_test_deviation])
    export_svg_heatmap(strat_strings, deviation_values, svg_name + "_heatmap_deviation.svg")
    mappings_values = np.array(
        [[cell if cell is not None else "" for cell in row] for row in sign_test_mappings])
    export_svg_heatmap(strat_strings, mappings_values, svg_name + "_heatmap_mappings.svg")

    # export bar charts for alignment, deviation and mappings:
    strat_strings = ["nechakhin_result", "zero-shot", "few-shot", "chain-of-thought", "combined", "optimized"]
    alignment_list = [row[0] for row in sums]
    alignment_list = [value / float(n) for value in alignment_list]
    deviation_list = [row[1] for row in sums]
    deviation_list = [value / float(n) for value in deviation_list]
    mappings_list = [row[2] for row in sums]
    mappings_list = [value / float(n) for value in mappings_list]
    export_svg_bar_chart("Average alignment values", 1, 5, strat_strings, alignment_list, svg_name + "_bar_chart_alignment.svg")
    export_svg_bar_chart("Average deviation values", 1, 5, strat_strings, deviation_list, svg_name + "_bar_chart_deviation.svg")
    export_svg_bar_chart("Average values of mappings", 0, math.ceil(max(mappings_list)), strat_strings, mappings_list, svg_name + "_bar_chart_mappings.svg")


def export_svg_bar_chart(label: str, min: int, max: int, strat_strings: list[str], chart_values: list[float], file_name: str) -> None:
    """
        Exports a horizontal bar chart to an SVG file.

        This function creates a horizontal bar chart with customizable labels,
        axis ranges, and chart values. It uses the Times New Roman font for all text
        and applies custom styling for a polished appearance. The chart is saved
        as an SVG file.

        Args:
            label (str): The label for the x-axis.
            min (int): The minimum value for the x-axis range.
            max (int): The maximum value for the x-axis range.
            strat_strings (list of str): Labels for the bars, one for each value in `chart_values`.
            chart_values (list of float): The values for the bar heights.
            file_name (str): The name of the SVG file to save the chart.

        Returns:
            None
        """
    plt.rcParams['font.family'] = 'Times New Roman'
    y_pos = np.arange(len(chart_values))
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.2)
    ax.barh(y_pos, chart_values, color="#005f50", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(strat_strings, fontsize=20)
    ax.set_xlabel(label, fontsize=20)
    ax.set_xlim(min, max)
    ax.set_xticks(range(min, max+1))
    ax.set_xticklabels(range(min, max+1), fontsize=20)
    ax.grid(axis='x', color='grey', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.savefig(file_name, format='svg', dpi=300)
    plt.close(fig)


def export_svg_heatmap(strat_strings: list[str], heatmap_values: list[list[str]], file_name: str) -> None:
    """
        Exports a heatmap to an SVG file.

        This function creates a square heatmap based on the provided values. Each cell
        is colored according to a p-value significance scale, with text annotations
        for each value. Gridlines and tick labels are configured for clarity, and the
        plot is styled using the Times New Roman font. The heatmap is saved as an SVG file.

        Args:
            strat_strings (list of str): Labels for the rows and columns of the heatmap.
            heatmap_values (list of list of str): A 2D list of values for each cell in the heatmap.
            file_name (str): The name of the SVG file to save the heatmap.

        Returns:
            None
        """
    plt.rcParams['font.family'] = 'Times New Roman'
    n = len(strat_strings)
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.15)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="grey", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(strat_strings, rotation=45, ha="right", fontsize=20)
    ax.set_yticklabels(strat_strings, fontsize=20)
    for i in range(n):
        for j in range(n):
            value = heatmap_values[i][j]
            color = cell_color(value) if value != "" else "white"
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=color))
            if value != "":
                ax.text(
                    j, i, f"{float(value):.3e}" if float(value) < 0.001 else f"{float(value):.3f}",
                    ha="center", va="center", color="black" if color != "#005f50" else "white",
                    fontsize=20
                )
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect("equal")
    plt.savefig(file_name, format="svg", bbox_inches="tight")
    plt.close(fig)


def cell_color(value: Union[str, None]) -> str:
    """
    Determines the cell color for a heatmap based on a p-value.

    This function maps p-values to specific colors to indicate significance levels:
    - Dark green (#005f50) for p ≤ 0.001
    - Medium green (#80afa8) for p ≤ 0.01
    - Light green (#ccdfdc) for p ≤ 0.05
    - White for p > 0.05 or invalid values

    Args:
        value (str or None): The p-value as a string. If empty or invalid, returns white.

    Returns:
        str: A color code (hex format) representing the p-value's significance.
    """
    if value == "" or value is None:
        return "white"
    try:
        value = float(value)
    except ValueError:
        return "white"
    if value <= 0.001:
        return "#005f50"
    elif value <= 0.01:
        return "#80afa8"
    elif value <= 0.05:
        return "#ccdfdc"
    return "white"


def postprocessing_response(string: str) -> list:
    """
    Processes an LLM response to extract and clean a list from its content.

    Args:
        string (str): The raw LLM response containing a list structure.

    Returns:
        list: A cleaned Python list extracted from the response.
    """
    if "[" not in string or "]" not in string:
        empty = list()
        return empty
    liste = "[" + string.split("[")[1].strip()
    liste = liste.split("]")[0].strip() + "]"
    liste = liste.replace("\n", " ")
    liste = liste.replace("  ", " ")
    liste = liste.replace("  ", " ")
    liste = liste.replace("  ", " ")
    return liste


def store_csv_data(table: dict) -> List[dict]:
    """
    Converts relevant data from a CSV file table into a list of dictionaries.

    Args:
        table (dict): A dictionary representation of the CSV file.

    Returns:
        list: A list of dictionaries representing publications.
    """
    publications = list()
    research_problems = table["research_problem"]
    orkg_properties_all = table["orkg_properties"]
    nechakhin_result = table["gpt_dimensions"]
    nechakhin_mappings = table["mappings"]
    nechakhin_alignments = table["alignments"]
    nechakhin_deviations = table["deviations"]
    publications = list()
    for i in range(len(research_problems)):
        publication = {"research_problem": research_problems[i], "orkg_properties": orkg_properties_all[i],
                       "nechakhin_result": nechakhin_result[i], "nechakhin_mappings": nechakhin_mappings[i],
                       "nechakhin_alignment": nechakhin_alignments[i], "nechakhin_deviation": nechakhin_deviations[i]}
        publications.append(publication)

    return publications


def dimensions_and_eval_to_json(publications: List[dict]) -> None:
    """
    Write publications list to a JSON file.

    Args:
        publications (List[Dict[str, Any]]): List of publications with the dimensions and evaluation results.

    Returns:
        None
    """
    while True:
        file_name = input("Please enter the file name to save the JSON data (ending with .json): ")
        try:
            with open(file_name, 'w') as file:
                json.dump(publications, file, indent=4)
                print(f"Data saved to {file_name}")
                break
        except Exception as e:
            print(f"Error writing to file {file_name}: {e}")


def match_dimension_to_orkg(dimension: str, api_orkg_properties: List[dict]) -> Union[dict,None]:

    """
    Attempts to match a dimension to an existing ORKG property using various matching strategies.

    Args:
        dimension (str): The dimension string to map.
        api_orkg_properties (list): A list of dictionaries representing ORKG properties.

    Returns:
        dict or None: The matching ORKG property or None if no match is found.
    """
    for p in api_orkg_properties:
        # 1
        if dimension.lower() == p["label"].strip().lower():
            return p
    for p in api_orkg_properties:
        # 2
        if " " in dimension:
            replace1 = dimension.replace(" ", "_")
            replace2 = dimension.replace(" ", "-")
            replace3 = dimension.replace(" ", "")
            if replace1.lower() == p["label"].strip().lower():
                return p
            if replace2.lower() == p["label"].strip().lower():
                return p
            if replace3.lower() == p["label"].strip().lower():
                return p
        # 3
        if "-" in dimension:
            replace1 = dimension.replace("-", "_")
            replace2 = dimension.replace("-", " ")
            replace3 = dimension.replace("-", "")
            if replace1.lower() == p["label"].strip().lower():
                return p
            if replace2.lower() == p["label"].strip().lower():
                return p
            if replace3.lower() == p["label"].strip().lower():
                return p
        # 4
        if "_" in dimension:
            replace1 = dimension.replace("_", "-")
            replace2 = dimension.replace("_", " ")
            replace3 = dimension.replace("_", "")
            if replace1.lower() == p["label"].strip().lower():
                return p
            if replace2.lower() == p["label"].strip().lower():
                return p
            if replace3.lower() == p["label"].strip().lower():
                return p
        # 5 trying with singular
        inf = inflect.engine()
        # singular_noun only works for single words, no compounds
        words = dimension.replace("-", " ").replace("_", " ").split()
        singular_words = [
            inf.singular_noun(word) if inf.singular_noun(word) else word for word in words
        ]
        compound = " ".join(singular_words)
        # literal comparison singular compound
        if compound.lower() == p["label"].strip().lower():
            return p
        # then the same as above with replace white space:
        if " " in compound:
            replace1 = compound.replace(" ", "_")
            replace2 = compound.replace(" ", "-")
            replace3 = compound.replace(" ", "")
            if replace1.lower() == p["label"].strip().lower():
                return p
            if replace2.lower() == p["label"].strip().lower():
                return p
            if replace3.lower() == p["label"].strip().lower():
                return p
        # 6 if nothing worked so far, return None
        return None


def save_api_orkg_properties_pickle() -> None:
    """
    Fetches ORKG properties from the API and saves them to a pickle file for future use.

    Returns:
        None
    """
    pickle_file = "api_orkg_properties.pickle"
    # If the pickle file already exists from a former run pf the script, it will be reused instead of recreated
    if os.path.exists(pickle_file):
        return
    # Fetch properties
    url = "https://incubating.orkg.org/api/predicates"
    api_orkg_properties = fetch_all_properties(url)
    # Save to pickle
    with open(pickle_file, "wb") as f:
        pickle.dump(api_orkg_properties, f)


def fetch_all_properties(url: str) -> List[dict]:
    """
    Retrieves all predicates (properties) from the ORKG API.
    See API Documentation: http://tibhannover.gitlab.io/orkg/orkg-backend/api-doc/#predicates-list

    Args:
        url (str): The base URL for the ORKG predicates API.

    Returns:
        list: A list of dictionaries containing predicate details (id, label, description).
    """
    all_properties = []
    page = 0
    page_size = 20  # http://tibhannover.gitlab.io/orkg/orkg-backend/api-doc/#predicates-list

    while True:
        response = requests.get(
            f"{url}",
            params={"page": page, "size": page_size},
            headers={
                "Content-Type": "application/json;charset=UTF-8",
                "Accept": "application/json"
            }
        )

        if response.status_code != 200:
            raise Exception(f"Failed to fetch page {page}: {response.status_code}")

        data = response.json()
        content = data.get("content", [])

        for predicate in content:
            all_properties.append({
                "id": predicate.get("id"),
                "label": predicate.get("label"),
                "description": predicate.get("description")
            })

        if data.get("last", True):
            break

        page += 1

    return all_properties


def preprocessing_dimensions(dimensions_to_match: List[str]) -> None:
    """
    Preprocesses LLM-extracted dimensions to clean and normalize them for matching to existing ORKG properties from
    the API. Concretely, handling dimensions containing ( and /

    Args:
        dimensions_to_match (list): A list of dimensions to preprocess.

    Returns:
        None
    """
    for dimension in dimensions_to_match:
        if "(" in dimension and ")" in dimension:
            new = dimension.split("(")[0].strip()
            dimensions_to_match.append(new)
        if "/" in dimension:
            new = dimension.split("/")[0].strip()
            dimensions_to_match.append(new)
            new = dimension.split("/")[1].strip()
            dimensions_to_match.append(new)


def match_dimensions_to_orkg(publication: dict) -> None:
    """
    Matches extracted dimensions of a publication to existing ORKG properties fetched from the API in the function
    save_api_orkg_properties_pickle().

    Args:
        publication (dict): A dictionary containing publication details and dimensions.

    Returns:
        None
    """
    # get all the over 10.000 ORKG properties that can be fetched from the API
    save_api_orkg_properties_pickle()
    with open("api_orkg_properties.pickle", 'rb') as file:
        api_orkg_properties = pickle.load(file)
    n = 0
    all_matches = list()
    try:
        if not isinstance(publication["nechakhin_result"], list):
            dimensions = eval(publication["nechakhin_result"])  # list e.g.: [context, method, process]
        else:
            dimensions = publication["nechakhin_result"]
        # dimensions stays the same, dimensions_to_match will be appended by cases with () and /
        dimensions_to_match = dimensions.copy()
        preprocessing_dimensions(dimensions_to_match)
        n += 1
        matches = list()  # for the matches (result of match_dimension_to_orkg)
        for dimension in dimensions_to_match:  # dimension e.g. research_problem
            match = match_dimension_to_orkg(dimension, api_orkg_properties)
            if match is not None:
                matches.append(match)
        all_matches.append(matches)
    except Exception as e:
        print(f"Exception in match_dimensions_to_orkg: {e}")
        print("n: " + str(n))
        print('publication["dimensions"]: ' + str(publication["dimensions"]))
        print("publication: " + str(publication))
        print("type(string): " + str(type(publication["nechakhin_result"])))
        print("string von eval(string): " + str(publication["nechakhin_result"]))
    for string in publication["dimensions"]:
        try:
            if not isinstance(string, list):
                dimensions = eval(string)  # list e.g.: [research_problem, context, method, process]
            else:
                dimensions = string
            # dimensions stays the same, dimensions_to_match will be appended by cases with () and /
            dimensions_to_match = dimensions.copy()
            preprocessing_dimensions(dimensions_to_match)
            n += 1
            matches = list() # for the matches (result of match_dimension_to_orkg)
            for dimension in dimensions_to_match: # dimension e.g. research_problem
                match = match_dimension_to_orkg(dimension, api_orkg_properties)
                if match is not None:
                    matches.append(match)
            all_matches.append(matches)
        except Exception as e:
            print(f"Exception in match_dimensions_to_orkg: {e}")
            print("n: " + str(n))
            print('publication["dimensions"]: ' + str(publication["dimensions"]))
            print("publication: " + str(publication))
            print("type(string): " + str(type(string)))
            print("string von eval(string): " + str(string))
    publication["orkg_matches"] = all_matches


def eval_match_dimensions_to_orkg(publications: List[dict]) -> None:
    """
    Evaluates the matching of extracted dimensions to existing ORKG properties.

    This function calculates statistics for matches across all publications
    and prints or optionally saves the details to a text file and a chart to a svg file. It also computes
    the average number of dimensions and matches for each prompting strategy.

    Args:
        publications (list): A list of dictionaries containing publication details,
                             dimensions, and their matches.

    Returns:
        None
    """
    print("Would you like to save the evaluation details to a txt-file and svg-chart?")
    print("A: Yes")
    print("B: No")
    svg_name = ""
    while True:
        choice = input("Please enter A or B: ")
        if choice.upper() == "A":
            file_bool = True
            file_name = input("Please enter a file name to store the evaluation details "
                              "(without the endings .txt or .svg): ")
            svg_name = file_name
            file_name = file_name + ".txt"
            break
        elif choice.upper() == "B":
            file_bool = False
            file_name = "file_name.txt"
            break
        else:
            print("Invalid input. Please enter 'A' or 'B'")

    with open(file_name, 'w', encoding='utf-8', errors="ignore") as file:
        if file_bool:
            dual_output = DualOutput(file)
            sys.stdout = dual_output  # "print" function will output the given strings both to the console and to
            # the given file
            count_dimensions_all = [0, 0, 0, 0, 0, 0]
            count_matches_all = [0, 0, 0, 0, 0, 0]
            p = 0
            for publication in publications:
                p += 1

                count_dimensions_one = [0, 0, 0, 0, 0, 0]
                n = 0
                if not isinstance(publication["nechakhin_result"], list):
                    dimensions = eval(publication["nechakhin_result"])  # list e.g.: [research_problem, context, method, process]
                else:
                    dimensions = publication["nechakhin_result"]
                count_dimensions_one[n] = len(dimensions)
                count_dimensions_all[n] += len(dimensions)
                n += 1
                for string in publication["dimensions"]:
                    if not isinstance(string, list):
                        dimensions = eval(string)  # list e.g.: [research_problem, context, method, process]
                    else:
                        dimensions = string
                    count_dimensions_one[n] = len(dimensions)
                    count_dimensions_all[n] += len(dimensions)
                    n += 1
                count_matches_one = [0, 0, 0, 0, 0, 0]
                n = 0
                for matches in publication["orkg_matches"]:
                    count_matches_one[n] = len(matches)
                    count_matches_all[n] += len(matches)
                    n += 1
                print("-" * 100)
                print("Publication " + str(p) + ": ")
                print("# Dimensions: " + str(count_dimensions_one))
                print("# Matches:   " + str(count_matches_one))
            print("-" * 100)
            print("-" * 100)
            print("Average number of matches to existing ORKG properties")
            print("-" * 100)
            print("Overall Number of Publications: " + str(len(publications)))
            strat_strings = ["nechakhin_result", "zero-shot", "few-shot", "chain-of-thought", "combined", "optimized"]
            all_proportion_matches_dimensions = list()
            for i in range(6):
                print("-" * 100)
                print("Prompting strategy: " + strat_strings[i])
                proportion_matches_dimensions = count_matches_all[i]/float(count_dimensions_all[i])
                all_proportion_matches_dimensions.append(proportion_matches_dimensions)
                print("Average number of dimensions:  " + str(round(count_dimensions_all[i]/float(len(publications)),2)))
                print("Average number of matches:     " + str(round(count_matches_all[i]/float(len(publications)),2)))
                print("Proportion matches/dimensions: " + str(round(proportion_matches_dimensions, 2),))
            if file_bool:
                # Restore the original stdout
                sys.stdout = dual_output.console
            if file_bool:
                export_svg_bar_chart("Proportion matches/dimensions", 0, 1, strat_strings,
                                     all_proportion_matches_dimensions, svg_name + "_bar_chart_matches.svg")


def to_int(string: str, replacement: int, index1: int, index2: int) -> int:
    """
    Converts a string to an integer, with error handling for non-integer inputs.

    This function attempts to convert a string to an integer. If a ValueError occurs, it returns
    a specified replacement value instead and prints an error message indicating the problematic index.
    This approach helps handle cases where the LLM outputs unexpected non-integer values.

    Args:
        string (str): The string to be converted to an integer.
        replacement (int): The integer to return in case of a ValueError.
        index1 (int): The first index used for identifying the error location in the data structure.
        index2 (int): The second index used for identifying the error location in the data structure.

    Returns:
        int: The converted integer, or the replacement value if conversion fails.
    """
    try:
        return int(string)
    except ValueError:
        print(f'ValueError at publication["eval"][{index1}][{index2}]')
        return replacement


def calc_sign_test(publications: List[dict], strat1: int, strat2: int) -> dict:
    """
    Calculates the p-values for the differences between the two prompting strategies (strat1 and strat2 referenced by
     the respective indices regarding the alignment, deviation, and mapping results using the Sign test.
     The hypothesis should be always that strat2 performs better than strat1.

    The Sign test is applied because:
    1. Paired samples are given (e.g., zero-shot vs. optimized strategies for the same publications).
    2. The values are measured on an ordinal scale.

    For each metric (alignment, deviation, and mappings):
    - Extracts the relevant values for the two prompting strategies.
    - Calculates the differences between the paired values.
    - Computes the number of positive and negative differences (ignoring zeros).
    - Applies the Sign test using `scipy.stats.binomtest` to determine if strat1 performs better (or worse)
      than strat2, based on predefined expectations.

    Args:
        publications (list): A list of dictionaries containing publication data, including evaluation metrics
                             under the "eval" key. Each "eval" key holds a list of lists, where:
                             - Index 0 contains zero-shot results.
                             - Index 4 contains optimized results.
        strat1 (int): Index int for strategy 1
        strat2 (int): Index int for strategy 2

    Returns:
        dict: A dictionary containing p-values for each metric:
            - "alignment" (float): p-value for alignment differences.
            - "deviation" (float): p-value for deviation differences.
            - "mappings" (float): p-value for differences of the number of mappings.
    """
    results = dict()
    strat1_alignment = [
        to_int(publication["eval"][strat1][0], 2, strat1, 0) for publication in publications
        if "eval" in publication and isinstance(publication["eval"], list) and len(publication["eval"]) >= 5
    ]
    # for alignment replacement with 2 if there is a ValueError as 2 is the average value of alignment through all
    # strategies
    strat2_alignment = [
        to_int(publication["eval"][strat2][0], 2, strat2, 0) for publication in publications
        if "eval" in publication and isinstance(publication["eval"], list) and len(publication["eval"]) >= 5
    ]
    # calculating differences
    differences = [strat2 - strat1 for strat2, strat1 in zip(strat2_alignment, strat1_alignment)]

    # sum up signs
    positive_differences = sum(1 for diff in differences if diff > 0)
    negative_differences = sum(1 for diff in differences if diff < 0)

    # n without difference of 0
    n = positive_differences + negative_differences

    # sign test with binomtest, alternative greater because optimized is supposed to have higher values than zero-shot
    if n > 0:
        result = binomtest(positive_differences, n=n, p=0.5, alternative='greater')
        p_value = result.pvalue
        results["alignment"] = p_value
    else:
        results["alignment"] = None
        # 2 deviation
    strat1_deviation = [
        to_int(publication["eval"][strat1][1], 4, strat1, 1) for publication in publications
        if "eval" in publication and isinstance(publication["eval"], list) and len(publication["eval"]) >= 5
    ]
    # for alignment replacement with 4 if there is a ValueError as 2 is the average value of alignment through all
    # strategies
    strat2_deviation = [
        to_int(publication["eval"][strat2][1], 4, strat2, 1) for publication in publications
        if "eval" in publication and isinstance(publication["eval"], list) and len(publication["eval"]) >= 5
    ]
    # calculating differences
    differences = [strat2 - strat1 for strat2, strat1 in zip(strat2_deviation, strat1_deviation)]

    # sum up signs
    positive_differences = sum(1 for diff in differences if diff > 0)
    negative_differences = sum(1 for diff in differences if diff < 0)

    # n without difference of 0
    n = positive_differences + negative_differences

    # sign test with binomtest, alternative less because optimized is supposed to have lower values than zero-shot
    if n > 0:
        result = binomtest(positive_differences, n=n, p=0.5, alternative='less')
        p_value = result.pvalue
        results["deviation"] = p_value
    else:
        results["deviation"] = None
        # 3 mappings
    strat1_mappings = [
        to_int(publication["eval"][strat1][2], 0, strat1, 2) for publication in publications
        if "eval" in publication and isinstance(publication["eval"], list) and len(publication["eval"]) >= 5
    ]
    # for alignment replacement with 0 if there is a ValueError as 2 is the average value of alignment through all
    # strategies
    strat2_mappings = [
        to_int(publication["eval"][strat2][2], 0, strat2, 2) for publication in publications
        if "eval" in publication and isinstance(publication["eval"], list) and len(publication["eval"]) >= 5
    ]
    # calculating differences
    differences = [strat2 - strat1 for strat2, strat1 in zip(strat2_mappings, strat1_mappings)]

    # sum up signs
    positive_differences = sum(1 for diff in differences if diff > 0)
    negative_differences = sum(1 for diff in differences if diff < 0)

    # n without difference of 0
    n = positive_differences + negative_differences

    # sign test with binomtest, alternative greater because optimized is supposed to have higher values than zero-shot
    if n > 0:
        result = binomtest(positive_differences, n=n, p=0.5, alternative='greater')
        p_value = result.pvalue
        results["mappings"] = p_value
    else:
        results["mappings"] = None
    return results


def main():
    """
    Main execution function for the script.

    This function performs the following steps:
    1. Reads gold standard and comparison data from a CSV file and processes it into a table.
    2. Asks the user which gpt-3.5-turbo model to use.
    3. Extracts and evaluates dimensions for each publication and optionally saves the results to files..
    4. Matches dimensions to existing ORKG properties URIs fetched from the API.
    5. Evaluates the matching and optionally saves the results to files.
    6. Saves the final data to a JSON file.

    The user is prompted to select the LLM model (gpt-3.5-turbo-0125 or gpt-3.5-turbo-1106) for processing
    and to provide filenames for saving results.

    Returns:
        None
    """

    load_dotenv()
    table = read_csv("orkg_properties_llm_dimensions_dataset_test.csv")
    publications = store_csv_data(table)

    llm = 10
    while llm == 10:
        llm_ab = input("Which LLM should be used?\nA: gpt-3.5-turbo-0125\nB: gpt-3.5-turbo-1106\n")
        if llm_ab.lower() == "a":
            llm = 0
        if llm_ab.lower() == "b":
            llm = 1
    print(f"{len(publications)} publications read in")
    for publication in publications:
        extract_dimensions_one(publication, llm)
    eval_dimensions_all(publications, llm)
    print_eval_details(publications)
    # publications to pickle (saving in case of occurring error during the following steps before saving to json)
    with open("publications.pickle", "wb") as f:
        pickle.dump(publications, f)
    # matching of dimensions to existing ORKG properties fetched from the API
    for publication in publications:
        match_dimensions_to_orkg(publication)
    eval_match_dimensions_to_orkg(publications)
    # final saving everything to json
    dimensions_and_eval_to_json(publications)


if __name__ == "__main__":
    main()
