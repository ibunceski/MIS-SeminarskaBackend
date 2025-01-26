import os
from datetime import datetime

import psycopg2


class DataStorage:
    def __init__(self, db_url=None):
        self.db_url = db_url or os.getenv('DB_URL', "postgresql://dians:dians123@localhost:9555/diansdb")
        self._initialize_db()

    def _initialize_db(self):
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS issuer_dates (
                            issuer TEXT PRIMARY KEY,
                            last_date DATE
                        )
                    """)

                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS issuer_data (
                            date DATE,
                            issuer TEXT,
                            avg_price TEXT,
                            last_trade_price TEXT,
                            max_price TEXT,
                            min_price TEXT,
                            percent_change TEXT,
                            turnover_best TEXT,
                            total_turnover TEXT,
                            volume TEXT,
                            PRIMARY KEY (date, issuer)
                        )
                    """)
                    conn.commit()
        except psycopg2.Error as e:
            print(f"Error initializing database: {e}")

    def load_data(self):
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT issuer, last_date FROM issuer_dates")
                    return {row[0]: row[1] for row in cursor.fetchall()}
        except psycopg2.Error:
            return {}

    def update_issuer(self, issuer, last_date):
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cursor:
                    date_str = last_date.strftime('%Y-%m-%d') if last_date else None
                    cursor.execute("""
                        INSERT INTO issuer_dates (issuer, last_date)
                        VALUES (%s, %s)
                        ON CONFLICT (issuer) DO UPDATE
                        SET last_date = EXCLUDED.last_date
                    """, (issuer, date_str))
                    conn.commit()
        except psycopg2.Error as e:
            print(f"Error updating issuer: {e}")

    def get_issuer_date(self, issuer):
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT last_date FROM issuer_dates
                        WHERE issuer = %s
                    """, (issuer,))
                    row = cursor.fetchone()
                    if row and row[0]:
                        return row[0]
                    return None
        except psycopg2.Error as e:
            print(f"Error retrieving issuer date: {e}")
            return None

    # def save_issuer_data(self, data_rows):
    #     try:
    #         with psycopg2.connect(self.db_url) as conn:
    #             with conn.cursor() as cursor:
    #                 cursor.executemany("""
    #                     INSERT INTO issuer_data (
    #                         date, issuer, avg_price, last_trade_price, max_price, min_price,
    #                         percent_change, turnover_best, total_turnover, volume
    #                     ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    #                     ON CONFLICT (date, issuer) DO NOTHING
    #                 """, data_rows)
    #                 conn.commit()
    #     except psycopg2.Error as e:
    #         print(f"Error saving issuer data: {e}")
    def save_issuer_data(self, data_rows):
        try:
            formatted_rows = [
                (
                    datetime.strptime(row[0], "%d.%m.%Y").strftime("%Y-%m-%d"),
                    *row[1:]
                )
                for row in data_rows
            ]

            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cursor:
                    cursor.executemany("""
                        INSERT INTO issuer_data (
                            date, issuer, avg_price, last_trade_price, max_price, min_price,
                            percent_change, turnover_best, total_turnover, volume
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (date, issuer) DO NOTHING
                    """, formatted_rows)
                    conn.commit()
        except ValueError as ve:
            print(f"Date format error: {ve}")
        except psycopg2.Error as e:
            print(f"Error saving issuer data: {e}")

    def get_all_data(self):
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cursor:
                    # cursor.execute("SELECT * FROM issuer_dates")
                    # issuer_dates = cursor.fetchall()

                    # Retrieve data from issuer_data table
                    cursor.execute("SELECT * FROM issuer_data")
                    issuer_data = cursor.fetchall()

                    return issuer_data
        except psycopg2.Error as e:
            print(f"Error retrieving data: {e}")
            return None, None

    def count_issuer_data_rows(self):
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) FROM issuer_data")
                    row_count = cursor.fetchone()[0]
                    return row_count
        except psycopg2.Error as e:
            print(f"Error counting rows in issuer_data table: {e}")
            return 0

    def get_by_issuer(self, issuer):
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT * FROM issuer_data WHERE issuer = %s
                    """, (issuer,))
                    rows = cursor.fetchall()
                    return rows
        except psycopg2.Error as e:
            print(f"Error retrieving data for issuer {issuer}: {e}")
            return []
