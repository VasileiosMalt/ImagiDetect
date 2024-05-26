
# ImagiDetect

ImagiDetect is an image classification system designed for social research on visual political data. It uses a pre-trained model and object-detection-based analysis for ethnographically informed categorization.

## Project Overview

This program was created for the ImagiDem project. The initial implementation for the develoment of the computational methodology of the project has been modernised (more recent models and python version are used). More information about the project can be found on the [ImagiDem website](https://csd.fi/imagidem/).

An article was published connected to the methodology used to develop this program:
> Maltezos, V., Luhtakallio, E., & Meriluoto, T. (2024). Bridging ethnography and AI: a reciprocal methodology for studying visual political action. International Journal of Social Research Methodology, 1–16. https://doi.org/10.1080/13645579.2024.2330057

## Project Structure (Same for each model)

- `classes.pkl`: Pickle file containing the class data.
- `classifier.pkl`: Pickle file containing the trained classifier. You will have to download that from [Hugging Face](Vasilis-Malt/ImagiDetect).
- `class_labels.npy`: Numpy file containing the class labels.
- `imagidetect.py`: Script to perform inference on a set of images.
- `predictions`: Directory where classified images are saved.
- `READ_ME.txt`: Brief description of the project and instructions.
- `requirements.txt`: List of required packages.
- `specific_requirements.txt`: Specific module versions used during testing.
- `supplementary_program.py`: Script for supplementary processing of classified images.

## Prerequisites

- Python 3.11.5

## Setup

1. **Install the required packages:**

   Install the necessary packages listed in `requirements.txt` and `specific_requirements.txt`.

   ```bash
   pip install -r requirements.txt
   ```

   For specific module versions:

   ```bash
   pip install -r specific_requirements.txt
   ```

2. **Place the inference (test) image folder in the current working directory:**

   Make sure the folder containing the images to be classified is placed in the same directory as the scripts.

## Usage

### Running the Inference Script

The inference script classifies images and optionally runs the supplementary program for further processing.

```bash
python imagidetect.py --test_data <path_to_test_data_folder> --prob_limit <probability_limit> [--no_supplementary]
```

- `--test_data`: Path to the folder containing the images to be classified.
- `--prob_limit`: Probability threshold for classification.
- `--no_supplementary`: Optional flag to skip running the supplementary program.

**Example:**

```bash
python imagidetect.py --test_data images --prob_limit 0.5
```

This command will classify the images in the `images` folder with a probability limit of 0.5 and run the supplementary program by default.

### Running the Supplementary Program

If you want to run the supplementary program separately, you can do so by:

```bash
python supplementary_program.py --dest_name <name_of_test_data_folder>
```

**Example:**

```bash
python supplementary_program.py --dest_name images
```

## Detailed Description

### Inference Script (`imagidetect.py`)

**Purpose:** Classifies images using a pre-trained classifier and extracts features from the images.  
**Output:** The classified images are organized into a directory structure based on their predicted classes and probabilities.

### Supplementary Program (`supplementary_program.py`)

**Purpose:** Further processes the classified images to organize them into specific categories and perform additional analysis.  
**Output:** The images are moved to subdirectories based on supplementary categorization rules.

## Notes

- Ensure that the test image folder is in the current working directory (same directory as the imagidetect.py script).
- The supplementary program will run by default after inference unless the `--no_supplementary` flag is provided.
- Tested Successfully In: Python 3.11.5

## Licence

The program is copyrighted by the University of Helsinki.

