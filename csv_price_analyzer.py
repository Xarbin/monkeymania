# CSV Price Analyzer - Add this to your MonkeyMania project as csv_price_analyzer.py

import pandas as pd
import numpy as np

def analyze_csv_prices(csv_path):
    """
    Analyze a CSV file to understand price data availability
    """
    df = pd.read_csv(csv_path)
    
    print(f"\n📊 Analyzing: {csv_path}")
    print(f"Total rows: {len(df)}")
    print("\n🔍 Price Column Analysis:")
    
    # Check various price columns
    price_columns = ['Pre-market Close', 'Price', 'Post-market Close']
    
    for col in price_columns:
        if col in df.columns:
            non_nan = df[col].notna().sum()
            valid_positive = (df[col] > 0).sum()
            avg_price = df[col][df[col] > 0].mean() if valid_positive > 0 else 0
            
            print(f"\n{col}:")
            print(f"  - Non-NaN values: {non_nan}/{len(df)} ({non_nan/len(df)*100:.1f}%)")
            print(f"  - Positive values: {valid_positive}/{len(df)} ({valid_positive/len(df)*100:.1f}%)")
            print(f"  - Average price: ${avg_price:.2f}")
            
            # Show some examples
            examples = df[df[col].notna() & (df[col] > 0)][['Symbol', col]].head(3)
            if not examples.empty:
                print(f"  - Examples:")
                for _, row in examples.iterrows():
                    print(f"    {row['Symbol']}: ${row[col]:.2f}")
    
    # Analyze rows with no valid pre-market price but have regular price
    if 'Pre-market Close' in df.columns and 'Price' in df.columns:
        no_pm_but_price = df[
            (df['Pre-market Close'].isna() | (df['Pre-market Close'] <= 0)) & 
            (df['Price'].notna() & (df['Price'] > 0))
        ]
        
        print(f"\n⚠️ Rows with no Pre-market Close but valid Price: {len(no_pm_but_price)}")
        if len(no_pm_but_price) > 0:
            print("Examples:")
            for _, row in no_pm_but_price.head(5).iterrows():
                print(f"  {row['Symbol']}: Price=${row['Price']:.2f}, Gap={row.get('Gap % 1 day', 0):.1f}%")
    
    # Find completely unpriceable rows
    unpriceable = df[
        (df['Pre-market Close'].isna() | (df['Pre-market Close'] <= 0)) & 
        (df['Price'].isna() | (df['Price'] <= 0))
    ]
    
    print(f"\n❌ Completely unpriceable rows: {len(unpriceable)}")
    if len(unpriceable) > 0:
        print("Examples:")
        for _, row in unpriceable.head(5).iterrows():
            print(f"  {row['Symbol']}: {row.get('Description', 'No description')[:50]}")
    
    return df

def fix_csv_prices(csv_path, output_path=None):
    """
    Create a fixed version of the CSV with better price handling
    """
    df = pd.read_csv(csv_path)
    
    # Create a new column for trading price
    df['Trading_Price'] = np.nan
    
    for idx, row in df.iterrows():
        # Priority 1: Pre-market Close
        if pd.notna(row.get('Pre-market Close', np.nan)) and row.get('Pre-market Close', 0) > 0:
            df.at[idx, 'Trading_Price'] = row['Pre-market Close']
        # Priority 2: Regular Price
        elif pd.notna(row.get('Price', np.nan)) and row.get('Price', 0) > 0:
            df.at[idx, 'Trading_Price'] = row['Price']
        # Priority 3: Estimate from gap
        elif pd.notna(row.get('Gap % 1 day', np.nan)) and row.get('Gap % 1 day', 0) != 0:
            # Assume a base price and adjust by gap
            base_price = 5.0  # Typical penny stock price
            gap = row['Gap % 1 day']
            df.at[idx, 'Trading_Price'] = base_price * (1 + gap / 100)
        else:
            # Mark as needing manual input
            df.at[idx, 'Trading_Price'] = -1  # Sentinel value
    
    # Save fixed CSV
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\n✅ Saved fixed CSV to: {output_path}")
    
    # Report statistics
    valid_prices = (df['Trading_Price'] > 0).sum()
    needs_manual = (df['Trading_Price'] == -1).sum()
    
    print(f"\n📊 Fixed CSV Statistics:")
    print(f"  - Valid trading prices: {valid_prices}/{len(df)} ({valid_prices/len(df)*100:.1f}%)")
    print(f"  - Needs manual input: {needs_manual}")
    
    return df

# Example usage
if __name__ == "__main__":
    # Analyze your CSV
    csv_path = "movers_pre6_4.csv"
    analyze_csv_prices(csv_path)
    
    # Create a fixed version
    fix_csv_prices(csv_path, "movers_pre6_4_fixed.csv")