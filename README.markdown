# 🌳 Decision Tree Classifier for Diabetes Prediction

## 📖 Overview
This project implements three decision tree algorithms—**ID3**, **C4.5**, and **CART**—to classify diabetes outcomes based on a provided dataset (`diabetes.csv`). The goal is to build, evaluate, and compare these models using preprocessed training and evaluation data. The project includes data preprocessing scripts to clean and prepare the dataset, followed by training and testing decision trees to predict whether a patient has diabetes. The implementation is done in MATLAB, focusing on accuracy and error metrics for model evaluation.

## 🎯 Objectives
- **Build Decision Trees**: Implement ID3, C4.5, and CART algorithms to classify diabetes outcomes.
- **Preprocess Data**: Clean and transform the dataset by removing unnecessary features and encoding categorical variables.
- **Evaluate Performance**: Calculate accuracy and error rates on the evaluation dataset.
- **Compare Algorithms**: Analyze the performance of ID3, C4.5, and CART based on accuracy and error metrics.

## ✨ Features
- **Three Algorithms**:
  - **ID3**: Uses information gain to select features, suitable for categorical data.
  - **C4.5**: Enhances ID3 with gain ratio to handle bias toward features with many values.
  - **CART**: Uses Gini index for binary splits, supporting numerical and categorical data.
- **Data Preprocessing**:
  - Removes `id` and `name` columns as they are irrelevant for classification.
  - Encodes `history_of_diabetes` and `has_diabetes` as binary (0/1).
  - Converts `status_weight` to numerical indices.
- **Model Evaluation**: Computes accuracy and error rates for predictions on the evaluation dataset.
- **Tree Visualization**: Prints the structure of each decision tree for interpretability.

## 🛠 Prerequisites
To run this project, you need:
- **MATLAB** (R2016a or later recommended).
- **Dataset Files**:
  - `diabetes.csv`: Training dataset.
  - `evaluation_diabetes.csv`: Evaluation dataset.
- **Dependencies**: No external MATLAB toolboxes are required; standard MATLAB functions are used.

## 📦 Installation
1. Clone or download the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <project-directory>
   ```
3. Ensure the dataset files (`diabetes.csv` and `evaluation_diabetes.csv`) are in the project directory or a `data` folder.
4. Open MATLAB and set the working directory to the project folder.

## 🚀 Usage
1. **Preprocess the Data**:
   - Run the preprocessing scripts to prepare the datasets:
     ```matlab
     process_train_data
     process_eval_data
     ```
   - This generates `processed_train_data.csv` and `processed_eval_data.csv`.
2. **Run the Decision Tree Models**:
   - Execute each algorithm script to build, display, and evaluate the tree:
     ```matlab
     id3_tree
     c45_tree
     cart_tree
     ```
3. **Output**:
   - Each script prints:
     - The decision tree structure.
     - Accuracy (e.g., `Accuracy: 85.50%`).
     - Error rate (e.g., `Error: 14.50%`).
   - Example output for `id3_tree`:
     ```
     Decision Tree:
     Node: Feature = status_weight
       Value = 1
         Leaf: Class = 0
       Value = 2
         Node: Feature = history_of_diabetes
     Accuracy: 82.30%
     Error: 17.70%
     ```

## 📊 Code Structure
- **Data Preprocessing**:
  - `process_train_data.m`:
    - Loads `diabetes.csv`.
    - Removes `id` and `name` columns.
    - Encodes `history_of_diabetes` and `has_diabetes` as binary (0/1).
    - Converts `status_weight` to numerical indices.
    - Saves output to `processed_train_data.csv`.
  - `process_eval_data.m`:
    - Same preprocessing steps for `evaluation_diabetes.csv`, saving to `processed_eval_data.csv`.
- **Decision Tree Algorithms**:
  - `id3_tree.m`:
    - Implements the ID3 algorithm using information gain.
    - Functions: `calculateEntropy`, `chooseBestFeature`, `buildID3Tree`, `printTree`, `predictFromTree`.
  - `c45_tree.m`:
    - Implements the C4.5 algorithm using gain ratio.
    - Functions: `calculateEntropy`, `calculateSplitInfo`, `chooseBestFeatureC45`, `buildC45Tree`, `printTree`, `predictFromTree`.
  - `cart_tree.m`:
    - Implements the CART algorithm using Gini index and binary splits.
    - Functions: `calculateGini`, `chooseBestFeatureCART`, `buildCARTTree`, `printTree`, `predictFromTree`.

## 🔍 Algorithm Comparison
| Algorithm | Splitting Criterion | Feature Selection | Split Type | Complexity |
|-----------|--------------------|-------------------|------------|------------|
| **ID3**   | Information Gain   | Categorical       | Multi-way  | O(N * F * L) |
| **C4.5**  | Gain Ratio         | Categorical       | Multi-way  | O(N * F * L) |
| **CART**  | Gini Index         | Numerical/Categorical | Binary  | O(N * F * V) |

- **N**: Number of samples, **F**: Number of features, **L**: Number of feature levels, **V**: Number of unique feature values.
- **ID3**: Simple but biased toward features with many levels.
- **C4.5**: Improves ID3 by using gain ratio to reduce bias.
- **CART**: Supports binary splits, suitable for both numerical and categorical data.

## 📈 Optional Visualization
To visualize the decision trees, you can extend the code to generate graphical outputs using MATLAB's plotting functions or external tools like Graphviz.
- Save the tree structure as a `.dot` file and render it with Graphviz.
  *Note: Add a visualization by generating a `.dot` file or using MATLAB's `graph` functions if desired.*
