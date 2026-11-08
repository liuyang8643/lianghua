"""Single SQLite setup shared by factor registry and factor-run records."""
import sqlite3
from pathlib import Path
def connect(path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    connection=sqlite3.connect(path)
    connection.row_factory=sqlite3.Row
    return connection
