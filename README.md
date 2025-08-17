📘 Machine Learning for Turbulence Modeling and Flow Optimization in Aerospace

Author: Y. Sai Goutham
Roll No.: 24M0037
Dataset Source: Johns Hopkins Turbulence Database (JHTDB)
Code Source: Partially developed by me and based on Kaggle Dataset

🧠 Overview

Turbulence modeling is fundamental to aerospace design and fluid dynamics simulations. This project combines Machine Learning (ML) and Computational Fluid Dynamics (CFD) tools to predict and analyze turbulence behavior based on real scientific data from JHTDB.

🎯 Objectives

Retrieve and process turbulence velocity data from the JHTDB

Perform Exploratory Data Analysis (EDA) on the dataset

Build a machine learning model to predict turbulence intensity

Visualize flow and turbulence using 2D and 3D plots

📦 Required Libraries

To run this project smoothly, install the required dependencies:

!pip install pyvista seaborn
!apt-get install -y xvfb libgl1-mesa-glx
!pip install givernylocal


If you're facing compatibility issues, run the following:

!pip uninstall -y numpy pandas pyvista seaborn
!pip install --no-cache-dir numpy==1.24.4 pandas==1.5.3 pyvista seaborn


⚠️ Note: Some packages like google-colab, xarray, and tensorflow may raise compatibility warnings. Ensure they are not critical to your pipeline or use isolated environments.

📚 Imported Libraries
# Standard Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pyvista as pv
import plotly.graph_objects as go

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# JHTDB Interaction
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData

# Jupyter Display
from IPython.display import display, Image

🧬 Dataset Configuration
🔹 Available Datasets:
SUPPORTED_DATASETS = [
  "isotropic1024coarse", "isotropic1024fine", "isotropic4096", "isotropic8192",
  "sabl2048high", "rotstrat4096", "mhd1024", "mixing",
  "channel", "channel5200", "transition_bl"
]

🔸 Setup Example:
dataset_title = input("Select dataset: ").strip()
dimension = int(input("Enter dimension (2 or 3): "))
auth_token = 'edu.jhu.pha.turbulence.testing-201406'
output_path = './giverny_output'
dataset = turb_dataset(dataset_title=dataset_title, output_path=output_path, auth_token=auth_token)


You can store and reuse objects using %store:

%store dataset
%store dataset_title
%store dimension

🌐 Grid & Velocity Data Query

Set up a resolution grid to control the volume of queried data:

if dimension == 3:
    nx, ny, nz = 16, 16, 16
    ...
else:
    nx, ny = 32, 32
    ...


Query velocity data:

velocity_raw = getData(dataset, 'velocity', 1.0, 'none', 'lag8', 'field', points)
velocity = np.array(velocity_raw, dtype=np.float32)
velocity = np.squeeze(velocity)


📝 Note: Exception handling is included to manage empty results or mismatched dimensions.

🧪 Machine Learning Pipeline

A typical ML pipeline includes:

Feature extraction from velocity data

Training using RandomForestRegressor

Evaluation using MSE and R² metrics

📍 Model building and training code will be included in a separate notebook/module (if not already).

📊 Visualization

You can generate 2D and 3D turbulence field visualizations using:

Matplotlib

Seaborn

PyVista for interactive 3D plots

Plotly for browser-based interactive charts

📁 File Structure (Example)
project/
│
├── README.md
├── turbulence_analysis.ipynb       # Main notebook
├── giverny_output/                 # Data output directory
└── models/                         # Trained model files (optional)

📌 Notes

Always validate dataset and grid size to avoid API overload.

If getData() returns empty, try modifying grid resolution.

Use %store and %store -r to persist variables across notebook sessions.