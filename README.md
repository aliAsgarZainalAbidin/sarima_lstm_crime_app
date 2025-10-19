# README for SARIMA-LSTM Crime Prediction Application

## Overview

The SARIMA-LSTM Crime Prediction Application is designed to forecast crime incidents using a hybrid model that combines Seasonal Autoregressive Integrated Moving Average (SARIMA) and Long Short-Term Memory (LSTM) networks. The application provides a user-friendly interface for data input, model training, and visualization of results.

## Project Structure

```
sarima_lstm_crime_app/
├── src/
│   └── crime_forecast/
│       ├── __init__.py
│       ├── app.py
│       ├── config.py
│       ├── pipeline.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── io.py
│       │   └── preprocess.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── sarima.py
│       │   └── lstm.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   ├── plotting.py
│       │   └── maps.py
│       └── gui/
│           ├── __init__.py
│           └── gradio_ui.py
├── tests/
│   ├── test_pipeline.py
│   └── test_models.py
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd sarima_lstm_crime_app
pip install -r requirements.txt
```

## Usage

1. **Data Input**: Upload your crime data in CSV or Excel format. The data should contain at least two columns: one for dates and another for crime counts.

2. **Configuration**: Adjust the parameters for the SARIMA and LSTM models as needed. You can specify the length of the test set, the horizon for forecasting, and other model parameters.

3. **Run the Application**: Execute the application using the following command:

```bash
python src/crime_forecast/app.py
```

4. **Interface**: The Gradio interface will launch in your web browser, allowing you to interact with the application, visualize results, and download predictions.

## Testing

Unit tests are provided to ensure the functionality of the pipeline and models. To run the tests, use:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.