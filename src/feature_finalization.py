
import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


# ==========================================================
# Configuration
# ==========================================================

@dataclass
class FeatureFinalizationConfig:

    train_data_path = os.path.join(
        "data", "processed", "train.csv"
    )

    test_data_path = os.path.join(
        "data", "processed", "test.csv"
    )

    selected_features_path = os.path.join(
        "artifacts", "selected_features.pkl"
    )

    final_train_path = os.path.join(
        "data", "final", "train_selected.csv"
    )

    final_test_path = os.path.join(
        "data", "final", "test_selected.csv"
    )


class FeatureFinalization:

    def __init__(self):
        self.config = FeatureFinalizationConfig()

    def initiate_feature_finalization(self):

        try:
            logging.info("Loading processed train and test datasets.")

            train_df= pd.read_csv(self.config.train_data_path)
            test_df= pd.read_csv(self.config.test_data_path)

            logging.info("Loading selected feature list.")

            selected_features=load_object(self.config. selected_features_path)

            train_selected= train_df[selected_features + [train_df.columns[-1]]]
            test_selected=test_df[selected_features + [test_df.columns[-1]]]



            os.makedirs(
                os.path.dirname(self.config.final_train_path),
                exist_ok=True
            )

            train_selected.to_csv(
                self.config.final_train_path,
                index=False
            )

            test_selected.to_csv(
                self.config.final_test_path,
                index=False
            )

            logging.info("Final datasets created successfully.")

            return train_selected, test_selected

        except Exception as e:
            raise CustomException(e, sys)
