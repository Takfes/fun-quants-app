import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get database name from .env
DATABASE_NAME = os.getenv("DATABASE_NAME")

# Delete the database
os.remove(DATABASE_NAME)
