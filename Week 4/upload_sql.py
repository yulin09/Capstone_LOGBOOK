import mysql.connector

# Define database connection parameters
host = '103.219.251.246'
user = 'braincor_ps01'
password = 'Bangkit12345.'
database = 'braincor_ps01'
sql_file_path = r"C:\Users\wiwiw\Downloads\Captone\ML\CustomerSegmentation\pos_7_3105.sql"

# Read the SQL file
with open(sql_file_path, 'r') as file:
    sql_script = file.read()

# Establish a database connection
conn = mysql.connector.connect(
    host=host,
    user=user,
    password=password,
    database=database
)

# Create a cursor object
cursor = conn.cursor()

# Execute the SQL script
for result in cursor.execute(sql_script, multi=True):
    print(f"Running query: {result}")

# Commit the transaction
conn.commit()

# Close the cursor and connection
cursor.close()
conn.close()

print("SQL script executed successfully.")
