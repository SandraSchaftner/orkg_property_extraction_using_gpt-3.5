# ORKG Properties Extractor

### Automated ORKG Properties Extraction and Evaluation Using GPT-3.5-Turbo

This Python script processes research problem descriptions from scientific publications to extract and evaluate properties using the Large Language Model (LLM) GPT-3.5-Turbo. It performs automated extraction, evaluation, and matching of the extracted properties to ORKG properties retrieved via the ORKG API.

## Features

The script follows these main steps:

1. **Data Preparation**: Reads and structures gold standard and comparison data from a CSV file.
2. **Properties Extraction**: Extracts dimensions from publications using various prompting strategies.
3. **Evaluation**: Assesses extracted properties for alignment, deviation, and mappings to ORKG properties (gold standard).
4. **Matching**: Maps extracted dimensions to ORKG property URIs retrieved from the API.
5. **Output**: Generates and optionally saves evaluation results to text and SVG files, and writes all data to JSON.

## Prerequisites

### Dependencies
Install required Python packages using:
```bash
pip install -r requirements.txt
```

### Required Files
- **requirements.txt**: Contains the script's dependencies.
- **.env**: Stores the OpenAI API key and organization ID.
- **dimensions_system_prompts.yaml**: Contains prompts for dimensions extraction.
- **dimensions_eval_system_prompts.yaml**: Contains prompts for evaluation of extracted dimensions.
- **orkg_properties_llm_dimensions_dataset_test.csv**: Dataset for evaluation, based on Nechakhin et al., 2024, with minor modifications.

## Usage

Run the script:
```bash
python orkg_properties_extraction.py
```

### Main Functionalities

The script includes the following core components:
- **Data Handling**:
  - Read and preprocess CSV datasets.
  - Save extracted data and evaluations to JSON.
- **Extraction and Evaluation**:
  - Extract dimensions from publication descriptions.
  - Evaluate alignment, deviation, and mappings with ORKG properties.
- **Visualization**:
  - Export results as horizontal bar charts and heatmaps in SVG format.
- **Matching**:
  - Match extracted dimensions to ORKG properties fetched via the API.

## Output

- Text and SVG files containing evaluation results.
- JSON file containing extracted dimensions, evaluations, and matches.

## Credits

This script is based on research by Nechakhin et al., 2024:

Research paper: V. Nechakhin, J. D’Souza, and S. Eger, „Evaluating Large Language Models for Structured Science Summarization in the Open Research Knowledge Graph,“ Information, vol. 15, no. 6. MDPI AG, p. 328, Jun. 05, 2024. doi: 10.3390/info15060328. (19.01.2025)

Gold standard data set: Vladyslav Nechakhin, Jennifer D’Souza (2024). ORKG Properties and LLM-Generated Research Dimensions Evaluation Dataset [Data set]. LUIS. https://doi.org/10.25835/6oyn9d1n. (19.01.2025)

## License

CC BY-SA 4.0

## Acknowledgment

The author gratefully acknowledges Nechakhin, D'Souza, and Eger for publishing the dataset, prompts, and experimental setup. This made it possible to replicate, understand, and build upon their study. I hope that through my ideas and findings presented in this paper, as well as my publicly available program code, prompts, and result files on GitHub, I can give something back to Nechakhin, D'Souza and Eger, and to their project.
