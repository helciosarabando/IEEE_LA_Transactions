## Instructions on How to Run this Project

**Project**: Convolutional and long short-time memory network configuration to predict the remaining useful life of rotating machinery
**Author:** Hélcio Ferreira Sarabando
**GitHub:** helciosarabando

This project was developed using Python 3.8 and all the dependencies for its proper execution are included in the "requirements.txt" file.

### Setting up the Environment on Windows

Create a virtual environment.

    py -3 -m venv venv 

Activate the virtual environment.

    .\venv\Scripts\activate

Upgrade the package manager.

    pip install --upgrade pip

Install the project dependencies.

    pip install -r .\requirements.txt

### Setting up the Environment on Linux

Create a virtual environment.

    python3 -m venv venv 

Activate the virtual environment.

    source venv/bin/activate

Upgrade the package manager.

    pip install --upgrade pip

Install the project dependencies.

    pip install -r ./requirements.txt

### Pre-processing

Run the STFT and wavelet calculation script.

    python STFT_Bearing_RP.py

Run the Wavelet calculation script.

    python WAVELET_Bearing_RP.py

### Processing

Run the CNN-LSTM script for learning.

    python CNN-LSTM_Bearing_Learn_V1.py

Run the CNN-LSTM script for testing.

    python CNN-LSTM Bearing_Test_V1.py

### Post-processing

Run the EWMA/SVR script for learning.

    python EWMA_SVR_curve_fit.py

Run the Regression script for learning.

    python REGRESSION_to_Learn_V1.py

Run the Regression script for testing.

    python REGRESSION_to_Test_V1.py

### Notes

- Pre-processed and processed data will be stored in the "DATA" folder.
- Results will be shown in figures ploted inline.