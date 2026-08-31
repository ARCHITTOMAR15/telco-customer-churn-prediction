import os
import pickle
import yaml
import sys
import joblib

from src.logger import logger
from src.exception import CustomException

## Function 1 Readin YAML COINFIGURATION
def read_yaml(file_path):
    try:
        with open(file_path,"r") as yaml_file:
          logger.info(f"Reading YAML  file from {file_path}")
          config=yaml.safe_load(yaml_file)
        logger.info("YAML file loaded successfully.")
        return config
    except Exception as e :
        logger.error("Failed to read YAML file.")
        raise CustomException(e, sys)

## Function 2 : Create Directories 
def create_directories(paths):
    try:
        for path in paths:
           os.makedirs(path,exist_ok=True)
           logger.info(f"Directory Created:{path}")
    except Exception as e:
        raise CustomException(e, sys)

## Function 3 for Save Objects 

def save_object(file_path,obj):
    try:
        directory=os.path.dirname(file_path)
        os.makedirs(directory,exist_ok=True)

        joblib.dump(obj,file_path)
        logger.info(f"Object saved at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

## function 4 load object s

def load_object(file_path):

    try:
        logger.info(f"Loading Object From {file_path}")
        obj=joblib.load(file_path)
        logger.info("Object Loaded Successfully")

        return obj
    except Exception as e:
        raise CustomException(e, sys)



