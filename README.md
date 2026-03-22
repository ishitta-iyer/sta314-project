# STA314 Project: Pet Facial Expression Classification

This repository contains our STA314 project work on classifying pet facial expressions into `Angry`, `Happy`, and `Sad`.

The final and best-performing approach in this repo is the tree-based modeling workflow in `ensemble_trees.ipynb`. If you want the notebook that represents the final model, start there.

## Repository Overview

### Main modeling files

- `combined_code.ipynb`
  A combined notebook that places the code from the tree-based workflow, logistic regression baseline, and MobileNetV2 workflow into a single file. The notebook is ordered with trees first, followed by logistic regression, then MobileNetV2, and includes markdown cells introducing each model section.

- `ensemble_trees.ipynb`
  The final notebook and strongest model in the repository. This notebook builds the tree-based pipeline, compares feature combinations, and develops the final tree-oriented approach used for the project conclusions.

- `MobilenetV2.ipynb`
  A transfer-learning notebook using `MobileNetV2` for image classification. This is part of the deep learning exploration, but it is not the final model we selected.

- `logisticregression.py`
  A baseline script that converts images to grayscale, resizes them, flattens them into vectors, and fits a logistic regression classifier. This serves as a simpler benchmark against the more advanced models.

- `eda.ipynb`
  Exploratory data analysis notebook for inspecting the dataset, image sizes, preprocessing choices, and dataset structure before training the main models.

## Data and project files

- `data/classification-of-pet-facial-expression/`
  The dataset directory containing the train and test image folders used by the notebooks and scripts.

- `data/classification-of-pet-facial-expression.zip`
  A zipped copy of the dataset.

- `LICENSE`
  Project license file.

- `.gitignore`
  Git ignore rules for local machine artifacts.



