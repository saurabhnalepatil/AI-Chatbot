from dotenv import load_dotenv
import os

load_dotenv()

# Access environment
RESTRICT_OPEN_ENDED_QUERIES = os.getenv('RESTRICT_OPEN_ENDED_QUERIES', 'False').lower() in ('true', '1', 't')
