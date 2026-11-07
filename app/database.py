import os

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine
from sqlmodel import Session, SQLModel

from app import db_models

load_dotenv()

connection_url = URL.create(
    "mysql+pymysql",
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "3306")),
    database=os.getenv("DB_NAME"),
)

engine = create_engine(connection_url, echo=False)


def create_db_tables():

    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
