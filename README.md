# VLM Evaluator for Image/Video Classification

**VLM Evaluator** is a proof-of-concept (POC) evaluation tool designed to quickly test state-of-the-art (SOTA) models on image and video classification tasks. This tool allows you to:

- Evaluate models on single file inputs or on an entire dataset.
- Enforce a required dataset folder structure for bulk evaluation.
- Run inference in parallel over many files (from 10 to thousands).
- Compute standard classification metrics (accuracy, F1 score, recall, etc.) using scikit‑learn’s classification report.
- Automatically fall back to the Gemini model when Qwen inference fails due to limited GPU VRAM (e.g., on smaller GPUs like T4 or L4).
- Provide a user-friendly web interface for quick testing and comparison of model outputs.
- Generate a CSV file of detailed predictions and evaluation metrics for further analysis or download.

This repository is meant as a quick POC evaluation tool for researchers and developers who want to rapidly test multiple models on a reasonable amount of data and draw useful insights.

---

## Table of Contents

- [Features](#features)
- [Dataset Handling](#dataset-handling)
- [Installation](#installation)
- [Usage](#usage)
  - [Single File Mode](#single-file-mode)
  - [Dataset Mode](#dataset-mode)
- [Evaluation Metrics](#evaluation-metrics)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)

---

## Features

- **Multi-Modal Support:**  
  Evaluate on image or video inputs for classification tasks.
  
- **Flexible Evaluation Modes:**  
  - **Single File Mode:** Upload one file and optionally provide the expected output.
  - **Dataset Mode:** Specify a local dataset directory (organized by class) for bulk evaluation.

- **Standardized Dataset Format:**  
  The tool expects your dataset to be organized like this:
  ```
  data/ class1/ file1.(jpg, png, mp4, avi) class2/ file2.(jpg, png, mp4, avi)
  ```
  - **Built-in Evaluation Metrics:**  
Uses scikit‑learn’s `classification_report` to calculate accuracy, F1 score, recall, and more.

- **GPU Safety:**  
Qwen inference is wrapped to catch Out‑of‑Memory errors; if such an error occurs, the tool automatically falls back to the Gemini model.

- **Parallel Processing:**  
Uses Python’s `ThreadPoolExecutor` to process multiple files concurrently for improved speed and scalability.

- **CSV Reporting:**  
Generates a CSV file with detailed predictions and evaluation metrics for further analysis or download.

- **Web UI:**  
A simple and user-friendly interface built with Bootstrap and jQuery for running evaluations.

---

## Dataset Handling

For bulk evaluation, the tool expects your dataset to be stored locally in a specific directory structure:

- **Image Classification:**  
Each subdirectory under the dataset folder represents a class label and contains image files (with `.png`, `.jpg`, or `.jpeg` extensions).

- **Video Classification:**  
Each subdirectory represents a class label and contains video files (with `.mp4` or `.avi` extensions).

When a dataset directory is provided via the web form, the tool:
1. Validates the dataset structure.
2. Automatically infers the ground truth from the folder name.
3. Processes all files in parallel and computes evaluation metrics using scikit‑learn.

---

## Installation

1. **Clone the Repository:**
 ```bash
 git clone https://github.com/yourusername/ai-model-evaluator.git
 cd ai-model-evaluator
```

2. **Set up an environment**
```
python3 -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

3. **Install deps**
```
pip install -r requirements.txt
```


## Usage
### Single File Mode
* Upload a single image or video file.
* Provide a prompt for the inference.
* Optionally, enter the expected output to compute evaluation metrics.
* The tool will run inference using the selected model(s) and return predictions along with evaluation metrics.
### Dataset Mode
* Enter the absolute path to your dataset directory in the "Dataset Directory" field.
* Ensure your dataset follows the required folder structure.
* The tool will load all files, infer the ground truth from folder names, run inferences in parallel, and compute evaluation metrics across the dataset.

## Evaluation metrics
For classification tasks, the tool uses SK learn's classification report.


## Running the application:
1. **Start the app**
```
  python app.py
```

2. **Open your browser**


3. **Use web-interface:**
```
   -> In single file mode, upload a file, enter your prompt and expected output.
   -> In dataset mode, enter the dataset directory path.
   -> Select one or more models (use Ctrl/Command for multiple selections).
   -> Click "Run Inference" to begin the evaluation.
```
## Project structure
```graphql
ai-model-evaluator/
├── app.py             # Main Flask application
├── templates/index.html       # Front-end HTML interface (in the templates folder if using render_template)
├── requirements.txt   # Required Python libraries
├── README.md          # This comprehensive documentation file
├── uploads/           # Directory for temporarily storing uploaded files
├── data/              # (Optional) Sample dataset folder (structure: data/<class>/<file>)
```
