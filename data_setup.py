import pandas as pd
import sqlite3
from pathlib import Path

def setup_imdb_database(csv_path: str, db_path: str = 'imdb_reviews.db'):
    """
    Set up SQLite database with IMDB reviews data.
    """
    print("Reading CSV file...")
    df = pd.read_csv(csv_path)
    
    expected_columns = {'review', 'sentiment'}

    if not expected_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {expected_columns}")
    
    print(f"Found {len(df)} reviews")
    
    print("Creating database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS imdb_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        review_text TEXT NOT NULL,
        sentiment TEXT NOT NULL
    )
    """)
    
    batch_size = 1000
    total_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
    
    print("Inserting data into database...")
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        
        batch_data = df.iloc[start_idx:end_idx]
        batch_data = batch_data.rename(columns={'review': 'review_text'})
        
        batch_data.to_sql('imdb_reviews', 
                         conn, 
                         if_exists='append', 
                         index=False)
        
        print(f"Processed batch {i+1}/{total_batches}")
    
    print("Creating index on sentiment column...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment ON imdb_reviews(sentiment)")
    
    cursor.execute("SELECT COUNT(*) FROM imdb_reviews")
    count = cursor.fetchone()[0]
    print(f"\nVerification: {count} reviews inserted into database")
    
    cursor.execute("""
    SELECT sentiment, COUNT(*) as count 
    FROM imdb_reviews 
    GROUP BY sentiment
    """)
    distribution = cursor.fetchall()
    print("\nSentiment distribution:")
    for sentiment, count in distribution:
        print(f"{sentiment}: {count}")
    
    conn.close()
    print("\nDatabase setup complete!")

if __name__ == "__main__":
    csv_file = 'IMDB Dataset.csv'
    
    if not Path(csv_file).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    setup_imdb_database(csv_file)