import psycopg2
import sys


conn_params = {
 
}

# Connect to the PostgreSQL database
try:
    connection = psycopg2.connect(**conn_params)
    cursor = connection.cursor()

    # SQL query to fetch IDs from the table
    sql_query = "SELECT id, details FROM news_dawn;"

    # Execute the SQL query
    cursor.execute(sql_query)

    # Fetch all rows from the result
    rows = cursor.fetchall()
    rows=rows[0:50]
    # print(rows)
    # cursor.execute("SELECT * from keywords;")
    # tables=cursor.fetchall()
    # print(tables)
    # # Process each row
    text_list = []
    for row in rows:
        label_list = []
        label_id = row[0]
        details = row[1]
        text_list.append(details)

    
except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL:", error)

finally:
    # Close the cursor and connection
    if connection:
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed.")