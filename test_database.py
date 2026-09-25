from api.db.base import Base
from api.db.database import engine


def test_database_connection():
    print("Creating database tables...")

    Base.metadata.create_all(bind=engine)

    print("Database connection OK")
    print("Tables created successfully")


if __name__ == "__main__":
    test_database_connection()