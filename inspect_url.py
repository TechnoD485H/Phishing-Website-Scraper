"""
Looks up a specific URL in labeled_dataset.csv and prints its full feature
row — useful for understanding why the model got a particular URL wrong.
"""
import sys
import pandas as pd
 
 
def inspect_url(url, filename='labeled_dataset.csv'):
    df = pd.read_csv(filename)
    row = df[df['url'] == url]
 
    if row.empty:
        print(f"Couldn't find '{url}' in {filename}.")
        print("Tip: paste the exact URL as it appeared in the mismatch output — trailing slashes matter.")
        return
 
    print(f"Features for: {url}\n")
    for col, val in row.iloc[0].items():
        print(f"  {col:25} {val}")
 
 
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 inspect_url.py "http://the-url-you-want-to-check.com/"')
        sys.exit(1)
 
    inspect_url(sys.argv[1])