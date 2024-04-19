# read file output.json and print the count of elements in the list
# Usage: python count.py
# Output: count of elements in the list
# Example: 10

import json
def hello():
    print("Hello, World!")

with open("output.json", "r") as f:
    data = json.load(f)

print(f"Count of \"output.json\" :{len(data)}")


with open("../test_llama.json", "r") as f:
    data = json.load(f)

print(f"Count of \"test_llama.json\" :{len(data)}")