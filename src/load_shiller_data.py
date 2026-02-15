import pandas as pd
import numpy as np
from pathlib import Path

def load_shiller_data():
    """Load and process Shiller IE data from CSV file."""

    # Define paths
    data_path = Path(__file__).parent.parent / 'data' / 'ie_data_only.csv'
    results_path = Path(__file__).parent.parent / 'results'
    results_path.mkdir(exist_ok=True)

    # Read CSV
    df = pd.read_csv(data_path)

    # Convert first column to datetime using Date_Fraction for end-of-month dates
    first_col = df.columns[0]
    
    # Drop rows where the first column is NaN (empty rows at end of file)
    df = df.dropna(subset=[first_col])
    
    # Rename columns early to clean up names with spaces
    # Strip whitespace from all column names
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'10y UST': 'GS10'})
    
    # Convert numeric columns to float (they may have been read as strings with commas)
    numeric_cols = ['SPX', 'D', 'E', 'CPI', 'GS10', 'SPX_real', 'Div_Real', 'SPX_TR_Real', 'E_Real', 'CAPE']
    for col in numeric_cols:
        if col in df.columns:
            # Remove commas from numbers before converting (e.g., "1,033.85" -> "1033.85")
            if df[col].dtype == 'object' or df[col].dtype == 'str':
                df[col] = df[col].astype(str).str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    year_month = df[first_col].astype(str).str.split('.', expand=True)
    df['year'] = year_month[0].astype(int)
    df['month'] = year_month[1].astype(int)

    # Use Date_Fraction to get actual date within the month
    days_in_month = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01') + pd.offsets.MonthEnd(0)
    df['Date'] = days_in_month

    # Set date as index
    df = df.set_index('Date')
    df = df.drop(columns=['year', 'month'])
    
    # Drop unnamed columns (empty columns from source CSV)
    unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    df = df.drop(columns=unnamed_cols)

    # Create SPX_TR column (nominal total return index)
    df['SPX_TR'] = df['SPX_TR_Real'] * df['CPI']

    # Calculate Bond monthly returns
    df['Bond_10y_TR_monthly'] = calculate_bond_return(df['GS10'].values)
    
    # Calculate Bond cumulative total return index (cumulative product of monthly returns)
    df['Bond_10y_TR'] = df['Bond_10y_TR_monthly'].cumprod()

    # Save to results folder
    output_path = results_path / 'shiller_data_processed.csv'
    df.to_csv(output_path)

    return df

def bond_dirty_price(coupon, maturity, yield_sa):
    """
    Computes the dirty price of a coupon bond given its coupon rate, maturity, and yield.

    Parameters:
    coupon : float - Annual coupon rate (e.g., 0.05 for 5%)
    maturity : float - Time to maturity in years (can be fractional)
    yield_sa : float - Annual yield to maturity (decimal)

    Returns:
    price : float - Dirty bond price per 100 face value
    """
    periods_per_year = 2
    n_periods = maturity * periods_per_year  # Total number of semi-annual periods (can be fractional)
    semi_annual_coupon = (coupon * 100) / periods_per_year  # Semi-annual coupon payment
    semi_annual_yield = yield_sa / periods_per_year
    
    # Create payment times in semi-annual periods (from n_periods down to next payment)
    payment_times = np.arange(n_periods, 0, -1)
    
    # Discount each coupon payment
    price = np.sum(semi_annual_coupon * (1 + semi_annual_yield) ** (-payment_times))
    
    # Add discounted principal at maturity
    price += 100 * (1 + semi_annual_yield) ** (-n_periods)
    
    return price

def calculate_bond_return(yields):
    """Calculate total return for 10yr par bond held for 1 month."""
    returns = np.zeros(len(yields))

    maturity = 10  # years
    holding_period = 1/12  # 1 month in years

    for i in range(len(yields) - 1):
        y_buy = yields[i] / 100  # Convert from percentage
        y_sell = yields[i + 1] / 100

        # Par bond: coupon rate equals initial yield
        coupon_rate = y_buy

        # Buy at par (price = 100)
        price_buy = 100

        # Remaining maturity after 1 month
        remaining_maturity = maturity - holding_period

        # Calculate dirty price at sale
        price_sell = bond_dirty_price(coupon_rate, remaining_maturity, y_sell)

        # Total return (no interim cash flows)
        returns[i] = price_sell / price_buy

    # Last period has no return
    returns[-1] = np.nan

    return returns

if __name__ == '__main__':
    df = load_shiller_data()
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(df.head())