from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from yt_chatbot import YTChatbot
import chromadb
"""
RQWpF2Gb-gU: But what is quantum computing? --> Tech tutorial for accuracy testing
jF9tKMJVu78: What Scientists Are Beginning to Find in the Bermuda Triangle --> Documentary for accuracy testing

wsqZfEBcTQ0: Mac Mini review --> opinion video for distinguishing fact vs opinion
SVa3H4I3g84: Influencers keep flaunting their wealth --> Commentary video for distinguishing fact vs opinion

HiyzzcuaAac: How Your Brain Works & Changes, Huberman Lab Essentials --> Technical Video for hallucinating Testing
HAoKJT3af7Y: Evaluate LLMs in Python with DeepEval --> Coding video for hallucinating Testing 
"""

"""---------- Retriever ----------"""
yt_chatbot = YTChatbot(video_id="HiyzzcuaAac")
retriever = yt_chatbot.create_retriever()
ret_chain = retriever | RunnableLambda(yt_chatbot.format_context)
res = ret_chain.invoke("Who is Andrew ?")
print(res)




"""---------- Chatbot initiation"""
# question = ""
# yt_chatbot = YTChatbot("HAoKJT3af7Y")
# yt_chatbot.create_retriever()
# answer = yt_chatbot.get_answer(question)
# print(answer)


""" ------------ Vector generation --------------- """
vector_store_path = "vector_stores"
# embedding = OpenAIEmbeddings(model="text-embedding-3-small")
# path = "/Users/rajsethi/PycharmProjects/ai-agent-testing/agents/rag_youtube_chatbot/transcripts"
# def split_text(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
#     return text_splitter.split_text(text)
#
# files = os.listdir(path)
# for file in files:
#     with open(f"{path}/{file}", "r") as f:
#         video_text = f.read()
#         splitted_text = split_text(video_text)
#         Chroma.from_texts(texts=splitted_text, embedding=embedding, collection_name=file.replace(".txt",""),
#                                      persist_directory=vector_store_path)
#         print(f"Done: {file}")
