
# ImagiDetect

ImagiDetect is an image classification system designed for social research on visual political data. It uses a pre-trained model and object-detection-based analysis for ethnographically informed image classification.

![main_classes]()
<img src="https://github.com/user-attachments/assets/113eea31-8e74-486e-8dc6-8d8c8e3f52b2" width="50%" height="50%" />

## Project Overview

This program was created for the ImagiDem project. The initial implementation for the develoment of the computational methodology of the project has been modernised (more recent models and python version are used). More information about the project can be found on the [ImagiDem website](https://csd.fi/imagidem/).

An article was published connected to the methodology used to develop this program:
> Maltezos, V., Luhtakallio, E., & Meriluoto, T. (2024). Bridging ethnography and AI: a reciprocal methodology for studying visual political action. International Journal of Social Research Methodology, 1–16. https://doi.org/10.1080/13645579.2024.2330057

## Project Structure (Same for each model)

- `classes.pkl`: Pickle file containing the class data.
- `classifier.pkl`: Pickle file containing the trained classifier. **ATTENTION** : You will have to download that from [Hugging Face](https://huggingface.co/Vasilis-Malt/ImagiDetect/tree/main) due to GitHub filesize limits. You should download the corresponding classifier according to the model's folder name and place it into that folder. 
- `class_labels.npy`: Numpy file containing the class labels.
- `imagidetect.py`: Script to perform inference on a set of images.
- `predictions`: Directory where classified images are saved.
- `READ_ME.txt`: Brief description of the project and instructions.
- `requirements.txt`: List of required packages.
- `specific_requirements.txt`: Specific module versions used during testing.
- `supplementary_program.py`: Script for supplementary processing of classified images.

## Classes

All classes:
Simple Selfies, Artificial Selfies, Complex Selfies, Animal Protest, Action, Bikers, Mask Selfies, Near Wall, Neutral People Sit, Protest Far, Protest Selfies, Artificial Crowds, Crowds, Artificial Groupie, Complex Couple, Cooperation, Groupies Sign, Protest Couple, Groupies, Perf Costumes, Perf Lay Down, Performances, Drawings, Book Cover, Dolls, Clothing, Protest Materials, Artificial Pollution, Artificial Sea, Artificial Building, Drawn Selfies, Globe, Map, Artificial Simple, Artificial Animal, Artificial Flat, Artificial Drought, Artificial Fires, Artificial Flat2, Artificial Flat3, Artificial Forest, Artificial Garbage, Artificial Graph, Artificial Air, Artificial Infra, Artificial Protest Images, Threat, Meeting Deliberation, Baby, Bags, Bicycle, Blank1, Blank2, Bottle Items, Brushes, Card Pattern, Cats, Flowers, Clothing2, Buildings, Computer, Dist Pattern1, Dogs, Empty Spaces, Feet, Food Item2, Animals, Furniture, Gardening, Hand Bag, Ice, Make Up, Mug2, Neutral Wall, Newspaper, Pavement Pattern, Phone, Scenery Tree, Ship, Shoe, Skies, SunSet, Surroundings1, Surroundings2, Transport, Trunk, Under Water, Scenery Field, Water, Cars, Food Items, Motor Cycles, Scenery Water, Neutral Images, Ambiguous, Hand Objects, Hand Protest, Mug, Pollution, Threat Chaos Fire, Wind Energy

The main ethnographically informed image categories are:
- Protest Selfies
- Crowds
- Groupies
- Performances
- Protest Materials
- Artificial Protest Images
- Threat
- Meeting Deliberation
- Neutral Images
- Ambiguous

The rest are subcategories belonging to one of the main ones.


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
- Tested Successfully in Python 3.11.5
- This categorisation's accuracy (83%) is tested on images scraped using hashtags related to politics and political participation. That means that, if applied to "global" data, you might (or might not) notice worse accuracy. 

## Licence

The program is copyrighted by the University of Helsinki.

