TRAINING DATA BUILDER

1. Copy data_entry.html and data_entry_server.py into your my-llm project root:

E:\DEV\my-llm\

2. Run PowerShell from the project root:

python data_entry_server.py

3. Your browser will open:

http://127.0.0.1:8000

4. Enter a User message and Assistant response.
5. Click "Add to Training Data".

The tool appends this format to:

data/raw/training.txt

<USER> User message <ASSISTANT> Assistant response <END>

No extra Python packages are required.
