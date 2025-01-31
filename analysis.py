import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import wraps
import time
from contextlib import contextmanager
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
            
            # Calculate daily returns
            self.df['Daily_Return'] = self.df['Close'].pct_change() * 100
            
            # Calculate volatility (20-day rolling standard deviation)
            self.df['Volatility'] = self.df['Daily_Return'].rolling(window=20).std()
            
            # Sort by date
            self.df = self.df.sort_values('Date')
            
            logging.info("Data loading and cleaning completed successfully")
            logging.info(f"Loaded {len(self.df)} rows of data")
            logging.info(f"Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
            
            return self.df
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            logging.error(f"Available columns: {list(self.df.columns) if self.df is not None else 'None'}")
            raise
    
    @timer_decorator
    def perform_analysis(self):
        """Perform comprehensive market analysis."""
        analysis_results = {
            'price_metrics': {
                'highest_close': self.df['Close'].max(),
                'lowest_close': self.df['Close'].min(),
                'average_close': self.df['Close'].mean(),
                'price_range': self.df['Close'].max() - self.df['Close'].min()
            },
            'volume_metrics': {
                'highest_volume': self.df['Shares Traded'].max(),
                'average_volume': self.df['Shares Traded'].mean(),
                'total_turnover': self.df['Turnover (₹ Cr)'].sum()
            },
            'volatility_metrics': {
                'highest_daily_return': self.df['Daily_Return'].max(),
                'lowest_daily_return': self.df['Daily_Return'].min(),
                'average_volatility': self.df['Volatility'].mean()
            },
            'trend_analysis': {
                'current_trend': 'Bullish' if self.df['Close'].iloc[-1] > self.df['Close'].iloc[-20:].mean() else 'Bearish',
                'price_momentum': self.df['Close'].iloc[-1] - self.df['Close'].iloc[-20]
            }
        }
        return analysis_results
    
    def create_visualizations(self, output_dir='./plots/'):
        """Create market analysis visualizations."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Price Movement with Volume
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Price subplot
        ax1.plot(self.df['Date'], self.df['Close'], linewidth=2, color='#1f77b4')
        ax1.set_title('NSE Index Price Movement', fontsize=14, pad=20)
        ax1.set_ylabel('Price (₹)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=10)
        
        # Add 20-day moving average
        ma20 = self.df['Close'].rolling(window=20).mean()
        ax1.plot(self.df['Date'], ma20, 'r--', label='20-day MA', alpha=0.7)
        ax1.legend()
        
        # Volume subplot
        volume_colors = np.where(self.df['Close'] >= self.df['Close'].shift(1), '#2ecc71', '#e74c3c')
        ax2.bar(self.df['Date'], self.df['Shares Traded'], color=volume_colors, alpha=0.7)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', labelsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}price_volume.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Daily Returns Distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(self.df['Daily_Return'].dropna(), bins=50, kde=True)
        plt.title('Distribution of Daily Returns', fontsize=14)
        plt.xlabel('Daily Return (%)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}returns_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Volatility Over Time
        plt.figure(figsize=(12, 6))
        plt.plot(self.df['Date'], self.df['Volatility'], color='#9b59b6', linewidth=2)
        plt.title('Market Volatility Over Time (20-day rolling std)', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Volatility', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', labelsize=10)
        plt.savefig(f'{output_dir}volatility.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_report(self):
        """Generate a comprehensive market analysis report."""
        analysis = self.perform_analysis()
        
        report = f"""
NSE Market Analysis Report
-------------------------
Analysis Period: {self.df['Date'].min().strftime('%d-%b-%Y')} to {self.df['Date'].max().strftime('%d-%b-%Y')}

Price Analysis:
- Highest Close: ₹{analysis['price_metrics']['highest_close']:,.2f}
- Lowest Close: ₹{analysis['price_metrics']['lowest_close']:,.2f}
- Average Close: ₹{analysis['price_metrics']['average_close']:,.2f}
- Price Range: ₹{analysis['price_metrics']['price_range']:,.2f}

Volume Analysis:
- Highest Daily Volume: {analysis['volume_metrics']['highest_volume']:,.0f} shares
- Average Daily Volume: {analysis['volume_metrics']['average_volume']:,.0f} shares
- Total Turnover: ₹{analysis['volume_metrics']['total_turnover']:,.2f} Cr

Market Volatility:
- Highest Daily Return: {analysis['volatility_metrics']['highest_daily_return']:,.2f}%
- Lowest Daily Return: {analysis['volatility_metrics']['lowest_daily_return']:,.2f}%
- Average Volatility: {analysis['volatility_metrics']['average_volatility']:,.2f}%

Current Market Trend:
- Trend Direction: {analysis['trend_analysis']['current_trend']}
- Price Momentum: ₹{analysis['trend_analysis']['price_momentum']:,.2f}
"""
        return report

def main():
    # Initialize analyzer with the provided data
    analyzer = NSEAnalyzer('stock_data.csv')
    
    try:
        # Load and clean data
        analyzer.load_and_clean_data()
        
        # Create visualizations
        analyzer.create_visualizations()
        
        # Generate and save report
        report = analyzer.generate_report()
        with open('nse_analysis_report.txt', 'w') as f:
            f.write(report)
        
        logging.info("Analysis completed successfully")
        
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()