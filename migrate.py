import psycopg2
from dotenv import load_dotenv
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Load environment variables from .env
load_dotenv()

# Fetch the SUPABASE_DB_URL from environment variables
DATABASE_URL = os.getenv("SUPABASE_DB_URL")


# Debugging: Ensure DATABASE_URL is loaded (avoid printing sensitive information)
if DATABASE_URL:
    logging.info("SUPABASE_DB_URL loaded successfully.")
else:
    logging.error("SUPABASE_DB_URL is not set. Please check your .env file.")
    sys.exit(1)  # Exit the script with a non-zero status

# Define the migration SQL
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS devcontainers (
  id SERIAL PRIMARY KEY,
  url VARCHAR,
  devcontainer_json TEXT,
  devcontainer_url VARCHAR,
  repo_context TEXT,
  tokens INTEGER,
  model TEXT,
  embedding TEXT,
  generated BOOLEAN,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

# Add migration to fix existing table if it was created with the wrong schema
ALTER_TABLE_SQL = """
DO $$ 
BEGIN
  -- Check if id column exists and doesn't have a sequence
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'devcontainers' 
    AND column_name = 'id'
    AND column_default IS NULL
  ) THEN
    -- Drop the existing primary key constraint
    ALTER TABLE devcontainers DROP CONSTRAINT IF EXISTS devcontainers_pkey;
    
    -- Drop the id column
    ALTER TABLE devcontainers DROP COLUMN IF EXISTS id;
    
    -- Add the id column with SERIAL (auto-increment)
    ALTER TABLE devcontainers ADD COLUMN id SERIAL PRIMARY KEY;
    
    RAISE NOTICE 'Table devcontainers updated: id column now uses SERIAL';
  END IF;
END $$;
"""

def main():
    connection = None  # Initialize connection variable

    try:
        # Connect to the PostgreSQL database using the connection string
        connection = psycopg2.connect(DATABASE_URL)
        connection.autocommit = True  # Enable autocommit mode
        logging.info("Connection to the database was successful.")

        # Create a cursor to execute SQL queries
        cursor = connection.cursor()

        # Execute the CREATE TABLE statement
        cursor.execute(CREATE_TABLE_SQL)
        logging.info("devcontainers table created (if it didn't exist already).")
        
        # Execute the ALTER TABLE statement to fix existing tables
        cursor.execute(ALTER_TABLE_SQL)
        logging.info("Migration completed: Table schema updated if needed.")

        # Optionally, verify the table creation
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
        """)
        tables = cursor.fetchall()
        logging.info(f"Current tables in 'public' schema: {tables}")

        # Close the cursor
        cursor.close()

    except psycopg2.OperationalError as e:
        logging.error("OperationalError: Could not connect to the database.")
        logging.error("Please check your SUPABASE_DB_URL and ensure the database server is running.")
        logging.error(f"Details: {e}")
        sys.exit(1)  # Exit the script with a non-zero status

    except Exception as e:
        logging.error(f"Error running migration: {e}")
        sys.exit(1)  # Exit the script with a non-zero status

    finally:
        if connection:
            connection.close()  # Close the connection if it was established
            logging.info("Database connection closed.")

if __name__ == "__main__":
    main()
