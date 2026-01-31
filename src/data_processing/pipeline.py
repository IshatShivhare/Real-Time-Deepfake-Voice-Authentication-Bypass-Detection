"""
Data Processing Pipeline
Runs all ETL steps sequentially.
"""
import sys
import logging
from src.utils.config_loader import get_config
from src.utils.logger import setup_logger

from src.data_processing.download_organize import organize_data
from src.data_processing.extract_features import run_extraction
from src.data_processing.prepare_final import run_preparation
# Assuming prepare_final has a main function or we need to expose one

logger = setup_logger("DataPipeline")

def run_pipeline(steps=None):
    if steps is None:
        steps = ['organize', 'extract', 'prepare']
        
    config = get_config()
    
    logger.info(f"Starting pipeline with steps: {steps}")
    
    if 'organize' in steps:
        logger.info("Running Step 1: Organization")
        if not organize_data(config):
            logger.error("Organization failed. Aborting.")
            return False
            
    if 'extract' in steps:
        logger.info("Running Step 2: Feature Extraction")
        run_extraction(config)
        
    if 'prepare' in steps:
        logger.info("Running Step 3: Final Preparation")
        run_preparation(config)
        
    logger.info("Pipeline completed successfully.")
    return True

if __name__ == "__main__":
    run_pipeline()
