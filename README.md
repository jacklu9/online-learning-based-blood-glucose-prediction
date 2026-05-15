# Online Learning Based Blood Glucose Prediction

This repository contains the code for **Learning-Based Alarm Systems for Hypoglycemia and Hyperglycemia Prevention**, a project focused on short-term blood glucose forecasting for type 1 diabetes decision support.

The model predicts future blood glucose over a **40-minute horizon** from recent glucose, insulin, and carbohydrate time-series data. The goal is to support artificial pancreas and alarm-system workflows by warning about possible hypoglycemia or hyperglycemia early enough for a controller or clinician-facing system to respond.

## Motivation

Model Predictive Control (MPC) and artificial pancreas systems depend on accurate forecasts of future blood glucose. This is difficult because glucose dynamics vary strongly across patients and can also change over time for the same patient. Traditional offline training often requires a personalized model trained on a large fixed dataset, which may be slow to obtain and may not adapt well once the patient behavior changes.

This project explores an **online learning** approach: the predictor is updated continuously as new samples arrive, so it can adapt to new patient-specific patterns while making rolling forecasts.

## What the project does

- Builds a PyTorch Lightning training loop for online blood glucose prediction.
- Uses an LSTM-based recurrent neural network for sequential time-series modeling.
- Predicts multiple future glucose steps recursively across the prediction horizon.
- Uses rolling train, validation, and test windows at each time step.
- Supports input features such as blood glucose, basal insulin, bolus insulin, and carbohydrate intake.
- Logs and saves experiment artifacts, including model weights, metrics, learner state, dataset scaling state, CSV exports, and plots.
- Includes Syne-Tune based hyperparameter tuning utilities.

## Model overview

The prediction model is implemented in [`src/models/controller.py`](src/models/controller.py). It uses:

- an LSTM layer to capture temporal dependencies in the lookback window;
- a fully connected layer to map hidden states to blood glucose predictions;
- recursive multi-step forecasting, where each predicted blood glucose value is fed back into the next future step.

The default configuration from the code uses:

- lookback window: `60` samples, corresponding to about 5 hours after 5-minute resampling;
- prediction horizon: `8` samples, corresponding to about 40 minutes;
- hidden dimension: `16`;
- number of LSTM layers: `1`.

## Online learning workflow

At each time step, the dataset loader constructs:

- a training batch from recent historical samples;
- a validation batch from the following held-out rolling window;
- a test sample at the current forecast time.

The learner then:

1. runs a forward pass on the training batch;
2. computes mean squared error plus scheduled weight decay;
3. updates model weights with Adam;
4. evaluates validation and test predictions without gradient updates;
5. records metrics and predictions for later plotting.

The learner also includes scheduled weight decay and stochastic weight averaging style parameter updates. See [`src/training_and_evaluation/learner.py`](src/training_and_evaluation/learner.py).

## Dataset

The project was developed around simulated type 1 diabetes patient data from the **UVA/Padova simulator**, which is commonly used in artificial pancreas research and has been accepted by the FDA as an alternative to some animal testing contexts.

Expected input data format:

```text
data/
  site_1/
    dataset.csv
  site_2/
    dataset.csv
```

Each `dataset.csv` should contain time-series columns for the selected input and output features. The default arguments expect:

- `BloodGlucose`
- `basal`
- `bolus`
- `CHO`

The dataset loader normalizes each feature, constructs lookback-plus-horizon input sequences, and stores scaling factors so that experiments can be reproduced or continued.

## Installation

This project was developed with Python 3.9.18 and PyTorch Lightning 1.4.9.

Create and activate an environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The pinned requirements include CUDA-related PyTorch packages. Depending on your machine, you may need to install a PyTorch build that matches your CPU/GPU environment.

## Running an experiment

Place the prepared dataset under `data/site_<id>/dataset.csv`, then run:

```bash
python main.py --site_id 1 --experiment_name example_run
```

Useful arguments:

```bash
python main.py \
  --site_id 1 \
  --lookback_window 60 \
  --prediction_horizon 8 \
  --input_features BloodGlucose basal bolus CHO \
  --output_features BloodGlucose \
  --hidden_dim 16 \
  --num_layers 1 \
  --learning_rate 0.001 \
  --plot 1
```

Experiment outputs are written under `experiments/<experiment_name>/`, including:

- `model.pth`
- `results.pickle`
- `learner_state.pickle`
- `dataset_state.pickle`
- saved arguments
- archived source scripts
- prediction and training-loss plots

## Hyperparameter tuning

[`tune.py`](tune.py) provides Syne-Tune integration. It expects a Python config file exposing a `configuration_space()` function.

Example:

```bash
python tune.py \
  --config_file path/to/config.py \
  --optimization_metric val_loss_mse \
  --optimizer "Random Search" \
  --max_wallclock_time 10800 \
  --max_num_trials_started 500 \
  --n_workers 4
```

## Repository structure

```text
.
|-- main.py                         # Main online training/evaluation entry point
|-- tune.py                         # Syne-Tune hyperparameter tuning entry point
|-- requirements.txt
`-- src
    |-- dataloader
    |   `-- csv_dataset.py          # Rolling-window CSV dataset construction
    |-- models
    |   `-- controller.py           # LSTM recursive predictor
    |-- training_and_evaluation
    |   |-- learner.py              # Online learner and metrics
    |   `-- loss.py                 # MSE loss wrapper
    |-- plots.py                    # Prediction and metric plotting
    `-- utils.py                    # Experiment, logging, saving, and scheduling utilities
```

## Future work

The project presentation highlights several natural next steps:

- build a larger universal training dataset instead of relying on one randomly selected training patient;
- test physics-guided neural networks by incorporating a pancreas model;
- integrate the predictor with an artificial pancreas controller;
- add safety constraints for cases where forecast uncertainty or prediction error could lead to harmful control actions.

## Disclaimer

This repository is a research project for blood glucose prediction. It is not a medical device and should not be used for clinical decision-making without appropriate validation, safety analysis, and regulatory review.
