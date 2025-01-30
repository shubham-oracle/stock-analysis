
import numpy as np
import pandas as pd
from functools import wraps
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Execution Timer Decorator
def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

class NSEAnalyzer:
    def __init__(self, data):
        self.raw_data = data
        self.df = None
        
    @timer_decorator
    def load_and_clean_data(self):
        """Load and clean the NSE dataset."""
        try:
            # Read the CSV file
            self.df = pd.read_csv(self.raw_data)
            
            # Clean column names: strip spaces and standardize
            self.df.columns = self.df.columns.str.strip()
            
            # Rename the date column if needed
            date_column = [col for col in self.df.columns if 'date' in col.lower()][0]
            self.df = self.df.rename(columns={date_column: 'Date'})
            
            # Convert Date to datetime
            self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d-%b-%Y')
            
            # Clean and convert numeric columns
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Shares Traded', 'Turnover (₹ Cr)']
            for col in numeric_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            return self.df
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            logging.error(f"Available columns: {list(self.df.columns) if self.df is not None else 'None'}")
            raise
        
    
def main():
    # Initialize analyzer with the provided data
    analyzer = NSEAnalyzer('stock_data.csv')
    
    try:
        # Load and clean data
        analyzer.load_and_clean_data()
        
        # Create visualizations

        # Generate and save report
    
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()