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
from Models.AEmodels import AutoEncoderCNN


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



#### 1. LOAD DATA
# 1.1 Patient Diagnosis



def load_cropped(folder_path, csv_path, patient_list, sample_size=200):
    # Cargar el CSV con pandas
    df = pd.read_csv(csv_path)
    
    # Convertir el CSV a un diccionario para un acceso rápido
    patient_metadata = {row['CODI']: row['DENSITAT'] for _, row in df.iterrows()}
    
    # Inicializar la estructura de datos para almacenar los datos de los pacientes seleccionados
    patients_data = []
    
    # Iterar sobre cada paciente en la lista dada
    for patient_id in patient_list:
        # Revisar si el paciente está en el diccionario de metadatos
        if patient_id in patient_metadata:
            # Definir el path de la carpeta del paciente
            patient_folder = os.path.join(folder_path, patient_id)
            
            # Obtener todas las imágenes .png dentro de la carpeta del paciente
            images = glob.glob(os.path.join(patient_folder, "*.png"))
            
            # Si el paciente tiene imágenes en su carpeta
            if images:
                # Mezclar la lista de imágenes
                random.shuffle(images)
                
                # Seleccionar una muestra de tamaño sample_size o menos si hay menos imágenes
                images_sample = random.sample(images, min(sample_size, len(images)))
                
                # Crear la entrada para el paciente con sus imágenes y metadatos
                patient_data = {
                    'patient_id': patient_id,
                    'densitat': patient_metadata[patient_id],
                    'images': images_sample
                }
                
                # Añadir la información del paciente a la lista de datos de pacientes
                patients_data.append(patient_data)
    
    return patients_data

# Ejemplo de uso
folder_path = "Cropped/"
csv_path = "PatientDiagnosis.csv"
patient_list = ["12345", "67890"]  # Lista de IDs de pacientes específicos
patients_data = load_cropped(folder_path, csv_path, patient_list)

# Imprimir un ejemplo de la estructura de datos
print(patients_data[:1])  # Muestra el primer paciente como ejemplo



# 1.2 Patches Data

#### 2. DATA SPLITING INTO INDEPENDENT SETS

# 2.0 Annotated set for FRed optimal threshold

# 2.1 AE trainnig set

# 2.1 Diagosis crossvalidation set

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

