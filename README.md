# Model pipeline 

The project houses codes that adhere to the azure ml pipelines. 

The code is split into 3 parts: 

* Preprocessing

* Training

* Validating 

The code uses the data created in the https://github.com/elasorama/create-electricity-features project. 

# Creating the virtual environment

The virtual environment is created using the requirements.txt file. To create the environment, run the following command:

```bash
python3.11 -m venv electricity-models
```

To activate it, run the following command:

```bash
# Bash
source electricity-models/bin/activate

# Powershell
.\electricity-models\Scripts\activate.ps1
```

To install the requirements, run the following command:

```bash
pip install -r requirements.txt
```

# Testing 

To run tests, run the command: 
    
```bash
pytest
```
