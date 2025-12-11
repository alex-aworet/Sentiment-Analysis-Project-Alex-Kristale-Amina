import os
import psycopg2
import logging


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        dbname=os.getenv("DB_NAME", "sentiment_logs"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
    )


def log_inference(text, sentiment, confidence):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO inference_logs (text, sentiment, confidence)
            VALUES (%s, %s, %s)
            """,
            (text, sentiment, confidence),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logging.warning(f"Failed to log prediction to DB: {e}")
