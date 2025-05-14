import aiomysql 
import os 
from dotenv import load_dotenv

load_dotenv()

async def get_connection():
    return await aiomysql.connect(
        host = os.getenv("DB_HOST"),
        port = 3306, 
        user = os.getenv("DB_USER"),
        password = os.getenv("DB_PASS"),
        db = os.getenv("DB_NAME"),
        autocommit = True
    )