

# Argus

### Overview
This project includes multiple deep learning models for training and testing on various datasets. You can switch models by modifying the model name in `main.py`. The project supports MNIST, CIFAR10, and ImageNet datasets, with logs automatically saved during runtime.

## Setup Instructions

### 1. Clone the Repository
First, download the project code:
```bash
git clone <repository_url>
cd <project_name>
```

### 2. Install Required Packages
The project requires several Python packages for running deep learning models, data processing, and logging. Install the required dependencies using the `requirements.txt` file. Ensure you have `pip` installed, then run the following command to install all necessary packages:
```bash
pip install -r requirements.txt
```

### 3. Prepare Data Folders
The project requires the datasets to be organized in a specific folder structure. In the project root directory, create a `data` folder to store your datasets:
```bash
mkdir data
```

- Download and place the **MNIST**, **CIFAR10**, and **ImageNet** datasets inside the `data` folder. Ensure each dataset is organized properly within `data` so the code can access it seamlessly.

### 4. Set Up Output and Log Directories
The output directory is used for saving results, and within it, a `log` folder is created to store runtime logs. To set this up, run the following commands:
```bash
mkdir -p output/log
```

### 5. Run the Project
To start training or testing with the default settings, run the main script:
```bash
python main.py
```

You can switch between models by modifying the `model name` variable in `main.py` to specify the desired model.

### Logs
Logs generated during runtime will be saved in the `output/log` folder.

---

**Note**: Replace `<repository_url>` with the actual URL of the GitHub repository and `<project_name>` with the name of your project.
```

