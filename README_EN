## GTLF Project Documentation

## Project Overview
GTLF (General Time Series Learning Framework) is a versatile time series learning framework focused on flood forecasting and other time series analysis tasks. This project integrates deep learning models with physical meta-learning approaches to enhance prediction accuracy during extreme events such as flood seasons.

## Key Features
- Supports multiple time series models including GRKU, LSTM, and GRU.
- Provides feature interpretation using SHAP.
- Implements meta-learning strategies to improve model generalization.
- Includes diverse data preprocessing and feature engineering tools.

## Technology Stack
- Python 3.x
- PyTorch
- SHAP
- FAISS
- NumPy, Pandas, Scikit-learn

## Project Structure
```
├── Config/              # Configuration files directory
├── src/                 # Source code directory
│   ├── analyzer/        # Analyzer modules
│   ├── data/            # Data processing modules
│   ├── model/           # Model definitions
│   ├── trainer/         # Training logic
│   └── utils/           # Utility classes
├── README.md            # Project documentation
└── config_schema.json   # Configuration file schema
```

## Configuration Guide

### Configuration File Structure
- `data`: Dataset configuration section
  - `default_dataset`: Default dataset configuration
  - `datasets`: List of specific datasets
- `models`: Model configuration section, containing specific configurations for various models
- `analyzers`: List of analyzer configurations
- `output`: Output path configuration

### Creating New Components Guide

#### Creating a New Trainer
1. Create a new trainer class file under `src/trainer/`, inheriting from `BaseModelTrainer`
2. Implement the `train` and `predict` methods
3. Add the new trainer configuration to the configuration file, specifying `trainer.type` as the new trainer class name

#### Create a New Model
1. Create a new model class file under the `src/model/` directory
2. Inherit from an appropriate base class (e.g., `nn.Module`)
3. Implement the model structure and forward propagation logic
4. Register the new model type in `ModelFactory`
5. Add a new model configuration to the configuration file, setting the `type` field to the new model class name

#### Create a new analyzer
1. Create a new analyzer class file in the `src/analyzer/` directory, inheriting from AnalyzerBase
2. Implement the `analyze` method and analysis logic
3. Register the new analyzer type in AnalyzerFactory
4. Add the new analyzer configuration to the `analyzers` list in the configuration file

#### Create a new data processor
1. Create a new data processor class file in the `src/data/` directory, inheriting from `DataProcessorBase`
2. Implement the data processing logic
3. Register the new processor type in `DataProcessorFactory`
4. Add the new processor configuration to the `processors` list in the dataset configuration
