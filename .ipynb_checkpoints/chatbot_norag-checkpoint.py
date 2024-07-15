import os
import getpass
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from apikey import key

os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your openai key: ")

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key="sk-proj-9mcWaluch6QkArjWoy6WT3BlbkFJ5G0D63tXBVOI3UQG6sKl"
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# messages = [
#     SystemMessage(content="Solve the math problem"),
#     HumanMessage(content="What is 81 divied by 9")
# ]

# result = llm.invoke(messages)
# print(f"answer from AI:{result.content}")

# result = llm.invoke("waht is 81 divided by 9")
# print('full result: ')
# print(result)
# print('content only: 0')
# print(result.content)
