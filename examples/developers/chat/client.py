"""Client - For servers and multi-user applications.

Primary API: talu.Client
Scope: Single
"""

from talu import Client

# Load model once
client = Client("Qwen/Qwen3-0.6B")

# Create chats for different users
alice = client.chat(system="You are helpful.")
bob = client.chat(system="You are a pirate.")

# Each user has their own conversation
response = alice("What is Python?")
print(f"Alice: {response}")
response = response.append("What is it used for?")
print(f"Alice: {response}")

response = bob("What is Python?")
print(f"Bob: {response}")
response = response.append("What is it used for?")
print(f"Bob: {response}")

# Cleanup: release model resources
client.close()

# Or use a context manager (auto-cleanup)
with Client("Qwen/Qwen3-0.6B") as client:
    chat = client.chat()
    response = chat("Hello!")

"""
Topics covered:
* client.ask
* client.shared.model
"""
