import psycopg2

dsn = "postgresql://ee547_user:password@localhost:5432/ee547_db"

print("Trying to connect with:", dsn)
conn = psycopg2.connect(dsn)
print("Connected OK!")
conn.close()
