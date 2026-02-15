"""
Download 3-month Treasury bill rate from FRED
This is used as the risk-free rate for Sharpe ratio and Merton allocation calculations
"""

import pandas as pd
import urllib.request
from pathlib import Path


def download_tb3ms():
    """
    Download 3-month Treasury bill rate from FRED
    Returns monthly data
    """
    # FRED data URL for 3-month Treasury bill rate (secondary market rate)
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS"
    
    try:
        # Download the data
        print("Downloading 3-month Treasury bill data from FRED...")
        response = urllib.request.urlopen(url)
        data = response.read()
        
        # Save to file
        data_path = Path(__file__).parent.parent / 'data'
        data_path.mkdir(exist_ok=True)
        
        output_file = data_path / 'TB3MS.csv'
        with open(output_file, 'wb') as f:
            f.write(data)
        
        # Read and process
        df = pd.read_csv(output_file)
        # Handle different possible column names
        if 'DATE' in df.columns:
            df = df.rename(columns={'DATE': 'Date'})
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        
        # Convert annual rate to monthly decimal
        # TB3MS is in percent per year, we need monthly decimal return
        df['RF_Monthly'] = (1 + df['TB3MS'] / 100) ** (1/12) - 1
        
        print(f"Downloaded TB3MS data: {len(df)} months")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"Saved to: {output_file}")
        
        return df
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Using fallback method...")
        return create_fallback_rf_rate()


def create_fallback_rf_rate():
    """
    Create fallback risk-free rate if download fails
    Use historical averages and approximations
    """
    # Load the Shiller data to get date range
    data_path = Path(__file__).parent.parent / 'results' / 'shiller_data_processed.csv'
    shiller_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Create approximate RF rate from GS10
    # Historically, 3-month T-bills average about 1-2% below 10-year Treasuries
    # But this varies by period
    
    df = pd.DataFrame(index=shiller_df.index)
    
    # Approximate: 3-month rate ≈ 10-year rate - term premium
    # Use a simple approximation
    gs10_annual = shiller_df['GS10'] / 100  # Convert percentage to decimal
    
    # Simple approximation: TB3M ≈ 0.7 * GS10 (rough historical relationship)
    # This captures that short rates are typically lower than long rates
    tb3m_annual = gs10_annual * 0.7
    
    # Convert to monthly 
    df['RF_Monthly'] = (1 + tb3m_annual) ** (1/12) - 1
    df['TB3MS'] = tb3m_annual * 100  # Back to percentage for reference
    
    print("Created fallback RF rate from GS10 approximation")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Save the fallback data
    output_file = Path(__file__).parent.parent / 'data' / 'TB3MS_fallback.csv'
    df.to_csv(output_file)
    print(f"Saved fallback to: {output_file}")
    
    return df


if __name__ == '__main__':
    # Try to download, fallback if it fails
    rf_data = download_tb3ms()
    
    if rf_data is not None:
        print("\nSample of risk-free rate data:")
        print(rf_data.head())
        print("\n...")
        print(rf_data.tail())
