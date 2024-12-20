# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:14:32 2024

Example of Main Steps for the Detection of HPilory using AutoEncoders for
the detection of anomalous pathological staining

Guides: 
    1. Split into train and test steps 
    2. Save trainned models and any intermediate result input of the next step
    
@authors: debora gil, pau cano
email: debora@cvc.uab.es, pcano@cvc.uab.es
Reference: https://arxiv.org/abs/2309.16053 

"""
# IO Libraries
import sys
import os
import pickle

# Standard Libraries
import numpy as np
import pandas as pd
import glob
import random

# Torch Libraries
from torch.utils.data import DataLoader
import gc
import torch


## Own Functions
from AutoEncoder.Models.AEmodels import AutoEncoderCNN
from AutoEncoder.Models.datasets import Standard_Dataset, Paired_Dataset


def AEConfigs(Config):
    
    if Config=='1':
        # CONFIG1
        net_paramsEnc['block_configs']=[[32,32],[64,64]]
        net_paramsEnc['stride']=[[1,2],[1,2]]
        net_paramsDec['block_configs']=[[64,32],[32,inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsEnc['block_configs'][-1][-1]
     

        
    elif Config=='2':
        # CONFIG 2
        net_paramsEnc['block_configs']=[[32],[64],[128],[256]]
        net_paramsEnc['stride']=[[2],[2],[2],[2]]
        net_paramsDec['block_configs']=[[128],[64],[32],[inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsEnc['block_configs'][-1][-1]
   
        
    elif Config=='3':  
        # CONFIG3
        net_paramsEnc['block_configs']=[[32],[64],[64]]
        net_paramsEnc['stride']=[[1],[2],[2]]
        net_paramsDec['block_configs']=[[64],[32],[inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsEnc['block_configs'][-1][-1]
    
    return net_paramsEnc,net_paramsDec,inputmodule_paramsDec


######################### 0. EXPERIMENT PARAMETERS
# 0.1 AE PARAMETERS
inputmodule_paramsEnc={}
inputmodule_paramsEnc['num_input_channels']=3

# 0.1 NETWORK TRAINING PARAMS

# 0.2 FOLDERS
folder_path_cropped_sample = "C:/Users/mirvi/Desktop/mii/UAB/4.1/PSIV2/detect mateicules/repte3/psiv-repte3/data/Cropped_sample"
csv_path_patintdiagnosis = "C:/Users/mirvi/Desktop/mii/UAB/4.1/PSIV2/detect mateicules/repte3/psiv-repte3/data/PatientDiagnosis.csv"
#### 1. LOAD DATA
# 1.1 Patient Diagnosis
def load_cropped(folder_path, csv_path, patient_list, sample_size=200):
    # Cargar el CSV con pandas
    df = pd.read_csv(csv_path)
    
    # Convertir el CSV a un diccionario para un acceso rápido
    patient_metadata = {row['CODI']: row['DENSITAT'] for _, row in df.iterrows()}
    
    # Inicializar la estructura de datos para almacenar los datos de los pacientes seleccionados
    patients_data = []
    
    # Iterar sobre cada paciente en la lista de IDs proporcionada
    for patient_id in patient_list:
        # Buscar la carpeta correspondiente en el directorio Cropped con el formato <ID>_*
        patient_folder_pattern = os.path.join(folder_path, f"{patient_id}_*")
        patient_folders = glob.glob(patient_folder_pattern)
        
        # Si existe alguna carpeta que coincide con el patrón <ID>_*
        if patient_folders:
            # Tomar la primera coincidencia (suponiendo que hay solo una carpeta por paciente)
            patient_folder = patient_folders[0]
            
            # Verificar que el paciente esté en el CSV
            if patient_id in patient_metadata:
                # Obtener todas las imágenes .png dentro de la carpeta del paciente
                images = glob.glob(os.path.join(patient_folder, "*.png"))
                
                # Si el paciente tiene imágenes en su carpeta
                if images:
                    # Mezclar la lista de imágenes
                    random.shuffle(images)
                    
                    # Seleccionar una muestra de tamaño sample_size o menos si hay menos imágenes
                    images_sample = random.sample(images, min(sample_size, len(images)))

                    # Binariar densidad
                    if patient_metadata[patient_id] == "NEGATIVA":
                        dens = 0
                    else:
                        dens = 1
                    
                    # Crear la entrada para el paciente con sus imágenes y metadatos
                    patient_data = {
                        'patient_id': patient_id,
                        'densitat': dens,
                        'images': images_sample
                    }
                    
                    # Añadir la información del paciente a la lista de datos de pacientes
                    patients_data.append(patient_data)
    
    return patients_data
# 1.2 Patches Data

#### 2. DATA SPLITING INTO INDEPENDENT SETS

# 2.0 Annotated set for FRed optimal threshold
patient_health_status = {}
def load_annotated(patient_id, image1, image2, adaptive_threshold=True):
    """
    Compara dos imágenes de un paciente usando MSE para determinar si está sano o no,
    basado en un umbral adaptativo entre las imágenes.

    Parameters:
    - patient_id: ID del paciente.
    - image1: Primera imagen (numpy array).
    - image2: Segunda imagen (numpy array).
    - adaptive_threshold: Si es True, calcula un umbral adaptativo basado en MSE promedio.

    Returns:
    - None, pero actualiza el diccionario `patient_health_status` con el ID del paciente y su estado de salud.
    """
    # Asegurarse de que las imágenes tengan el mismo tamaño
    if image1.shape != image2.shape:
        raise ValueError("Las imágenes deben tener el mismo tamaño")
    
    # Calcular el MSE entre las dos imágenes
    mse = np.mean((image1 - image2) ** 2)
    
    # Umbral adaptativo (por ejemplo, ajustar con estadísticas previas)
    if adaptive_threshold:
        threshold = mse * 1.2  # Umbral dinámico que se ajusta en función del MSE promedio observado
    else:
        threshold = 0.01  # Umbral fijo, para pruebas iniciales
    
    # Decidir el estado de salud del paciente según el umbral
    is_healthy = mse < threshold
    health_status = "sano" if is_healthy else "no sano"
    
    # Almacenar el resultado en el diccionario
    patient_health_status[patient_id] = health_status
    print(f"Paciente {patient_id}: {health_status} (MSE: {mse}, Umbral: {threshold})")
# 2.1 AE trainnig set y Diagosis crossvalidation set
def create_training_set_kfold(folder_path, csv_path, k=10, sample_size=200):
    # Cargar la lista de IDs de pacientes del CSV
    df = pd.read_csv(csv_path)
    patient_list = df['CODI'].unique().tolist()
    
    # Cargar los datos usando la función load_cropped
    patients_data = load_cropped(folder_path, csv_path, patient_list, sample_size)
    
    # Preparar el KFold para dividir los datos
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    for fold, (train_index, val_index) in enumerate(kf.split(patients_data)):
        # Crear conjuntos de entrenamiento y validación
        train_data = [patients_data[i] for i in train_index]
        val_data = [patients_data[i] for i in val_index]
        
        # Crear DataLoaders para entrenamiento y validación
        train_loader = DataLoader(TrainingDataset(train_data), batch_size=32, shuffle=True)
        val_loader = DataLoader(TrainingDataset(val_data), batch_size=32, shuffle=False)

class TrainingDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        images = [torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) for image in item['images']]
        return torch.stack(images), item['densitat']
    
#### 3. lOAD PATCHES

### 4. AE TRAINING

# EXPERIMENTAL DESIGN:
# TRAIN ON AE PATIENTS AN AUTOENCODER, USE THE ANNOTATED PATIENTS TO SET THE
# THRESHOLD ON FRED, VALIDATE FRED FOR DIAGNOSIS ON A 10 FOLD SCHEME OF REMAINING
# CASES.

# 4.1 Data Split


###### CONFIG1
Config='1'
net_paramsEnc,net_paramsDec,inputmodule_paramsDec=AEConfigs(Config)
model=AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                     inputmodule_paramsDec, net_paramsDec)
# 4.2 Model Training

# Free GPU Memory After Training
gc.collect()
torch.cuda.empty_cache()
#### 5. AE RED METRICS THRESHOLD LEARNING

## 5.1 AE Model Evaluation

# Free GPU Memory After Evaluation
gc.collect()
torch.cuda.empty_cache()

## 5.2 RedMetrics Threshold 

### 6. DIAGNOSIS CROSSVALIDATION
### 6.1 Load Patches 4 CrossValidation of Diagnosis

### 6.2 Diagnostic Power

