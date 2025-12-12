# Neural and Evolutionary Learning: Models Benchmarking

**NOVA Information Management School**  
**Academic Year: 2024/2025**

**Course Instructors:**  
Karina Rebuli & Leonardo Vanneschi

**Authors:**  
- Iris Moreira
- Rafael Borges
- Leonardo di Caterina

---

## Overview

This repository contains a comprehensive benchmarking study that compares the performance of Genetic Programming families against Neural Network families on a regression task. The primary objective is to predict crude protein weight in chickens using five distinct machine learning approaches, evaluated through Root Mean Square Error (RMSE) as the primary performance metric.

The models compared in this study include three variants from the Genetic Programming family: Standard Genetic Programming (GP), Geometric Semantic Genetic Programming (GSGP), and SLIM-GSGP (a more efficient implementation of GSGP). These are benchmarked against two Neural Network-based approaches: a Standard Neural Network (NN) with hyperparameter optimization and NeuroEvolution of Augmenting Topologies (NEAT), which evolves both the network topology and weights simultaneously.

This research provides insights into the relative strengths and limitations of evolutionary computation versus gradient-based learning methods, with particular attention to model generalization, computational efficiency, and susceptibility to overfitting.

---

## Repository Structure

The repository is organized to facilitate reproducibility and understanding of the complete experimental pipeline. The workflow progresses from initial data exploration through individual model training to comprehensive statistical comparison.

### Notebooks

The notebooks should be executed in the following sequential order to replicate the complete analysis:

**Data Exploration and Preparation:**
- `0_EDA.ipynb` - Performs exploratory data analysis, examining feature distributions, correlations, and potential outliers in the chicken protein dataset.
- `1_Data_Preprocessing.ipynb` - Implements data cleaning, feature engineering, normalization, and prepares the dataset for model training.

**Individual Model Training:**
- `2_main_GP.ipynb` - Trains and evaluates the Standard Genetic Programming model using nested cross-validation.
- `3_main_GSGP.ipynb` - Implements Geometric Semantic Genetic Programming with semantic-aware operators.
- `4_main_SLIM_GSGP.ipynb` - Applies the SLIM variant of GSGP, which uses a more memory-efficient representation.
- `5_main_NN.ipynb` - Performs grid search hyperparameter optimization for the Standard Neural Network architecture.
- `6_main_NEAT.ipynb` - Executes the NEAT algorithm, allowing the network topology to evolve alongside connection weights.

**Comprehensive Analysis:**
- `7_main_all_models.ipynb` - Aggregates results from all models, performs statistical significance testing, and generates comparative visualizations including RMSE distributions and computational efficiency metrics.

### Utils

Supporting utility modules provide reusable functionality across the experimental pipeline:

- `prep_data.py` - Contains functions for data loading, splitting, scaling, and preparation for different model types.
- `grid_search.py` - Implements nested cross-validation with grid search for hyperparameter optimization.
- `visualization_funcs.py` - Provides standardized plotting functions for performance comparisons, error distributions, and training curves.
- `NEAT_utils.py` - Includes helper functions specific to NEAT configuration, fitness evaluation, and network visualization.
- `NN_utils.py` - Contains neural network architecture definitions, training loops, and evaluation utilities.

---

## Installation & Requirements

This project requires Python 3.12 or higher. To set up the necessary dependencies, execute the following command in your terminal:

```bash
pip install -r requirements.txt
```

**Important:** The SLIM-GSGP implementation requires the `slim_gsgp` library developed by the DALabNOVA research group. If this library is not included in the requirements file, you must install it directly from the GitHub repository using:

```bash
pip install git+https://github.com/DALabNOVA/slim.git
```

This ensures that the SLIM-GSGP variant can be properly executed with its optimized memory representation for geometric semantic operations.

---

## Usage Instructions

To replicate the complete experimental workflow, follow these steps in order:

**Step 1: Data Preparation**  
Begin by running the exploratory data analysis and preprocessing notebooks. The `0_EDA.ipynb` notebook will help you understand the dataset characteristics, while `1_Data_Preprocessing.ipynb` will create the cleaned and normalized dataset required for all subsequent model training.

**Step 2: Individual Model Training**  
Execute notebooks 2 through 6 in sequential order. Each notebook implements nested cross-validation to ensure robust hyperparameter selection and unbiased performance estimation. The outer cross-validation loop provides performance estimates on held-out test sets, while the inner loop optimizes model-specific hyperparameters. Note that these notebooks can be computationally intensive, particularly the neural network grid search and NEAT evolution, so consider running them on appropriate hardware or adjusting population sizes and generation counts if needed.

**Step 3: Comparative Analysis**  
Finally, run `7_main_all_models.ipynb` to aggregate all results, perform statistical significance testing (such as paired t-tests or Wilcoxon signed-rank tests), and generate comprehensive visualizations comparing model performance across all cross-validation folds. This notebook produces the key findings that inform the conclusions of the study.

---

## Key Results & Findings

The experimental results reveal several important insights into the comparative performance of evolutionary and neural network-based approaches for this regression task:

**Overall Performance:** Neural Network-based models (both Standard NN and NEAT) significantly outperformed all Genetic Programming variants in terms of RMSE. The neural networks demonstrated superior generalization capability and achieved lower prediction errors on held-out test data across all cross-validation folds.

**Overfitting in Genetic Programming:** Both GSGP and SLIM-GSGP exhibited a pronounced tendency toward overfitting. Despite their theoretical advantages in semantic space exploration, these models achieved substantially better training performance than test performance, suggesting that the semantic operators may have led to overly complex solutions that captured noise rather than underlying patterns in the data.

**Computational Efficiency:** NEAT demonstrated a notable advantage in computational efficiency compared to the exhaustive grid search approach used for the Standard Neural Network. Despite exploring both topology and weight spaces simultaneously, NEAT's evolutionary search proved more efficient than systematically evaluating all hyperparameter combinations, while achieving statistically equivalent predictive performance.

**Statistical Equivalence:** While both neural network approaches significantly outperformed GP variants, the difference between Standard NN and NEAT was not statistically significant according to paired comparison tests. This suggests that NEAT's neuroevolutionary approach can match the performance of carefully tuned feedforward networks while requiring less domain knowledge for architecture design.

These findings suggest that for similar regression problems with moderate-sized datasets, neural networks provide more reliable solutions than genetic programming approaches, with NEAT offering an attractive balance between performance and computational cost.

---

## References & Credits

This project makes use of the SLIM library for efficient Geometric Semantic Genetic Programming, developed and maintained by the DALabNOVA research group at NOVA Information Management School. We extend our gratitude to the development team for making this implementation publicly available.

**SLIM Library:**  
Repository: [https://github.com/DALabNOVA/slim](https://github.com/DALabNOVA/slim)  
Maintained by: DALabNOVA Team

For questions or additional information about this project, please contact the authors through the course instructors.

---

**License:** This project is developed for academic purposes as part of the Neural and Evolutionary Learning course at NOVA IMS.
