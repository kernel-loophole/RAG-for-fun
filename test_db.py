# import psycopg2
# import sys


# conn_params = {
 
# }

# # Connect to the PostgreSQL database
# try:
#     connection = psycopg2.connect(**conn_params)
#     cursor = connection.cursor()

#     # SQL query to fetch IDs from the table
#     sql_query = "SELECT id, details FROM news_dawn;"

#     # Execute the SQL query
#     cursor.execute(sql_query)

#     # Fetch all rows from the result
#     rows = cursor.fetchall()
#     rows=rows[0:50]
#     # print(rows)
#     # cursor.execute("SELECT * from keywords;")
#     # tables=cursor.fetchall()
#     # print(tables)
#     # # Process each row
#     text_list = []
#     for row in rows:
#         label_list = []
#         label_id = row[0]
#         details = row[1]
#         text_list.append(details)

    
# except (Exception, psycopg2.Error) as error:
#     print("Error while connecting to PostgreSQL:", error)

# finally:
#     # Close the cursor and connection
#     if connection:
#         cursor.close()
#         connection.close()
#         print("PostgreSQL connection is closed.")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

messages = [
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
