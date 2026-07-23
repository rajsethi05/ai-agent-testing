from agents.rag_youtube_chatbot.yt_chatbot import YTChatbot

yt = YTChatbot("SVa3H4I3g84")
retriever = yt.create_retriever()
# res= yt.get_answer("How does shrinkflation reduce product size while maintaining or increasing retail prices?")

results = retriever.invoke("How does shrinkflation reduce product size while maintaining or increasing retail prices?")
result = "--".join(res.page_content for res in results)
print(result)
print(results)


"""
Exactly. The test flow is:

Take input from golden dataset
Pass it to your RAG retriever
Get the retrieved chunks
Verify those retrieved chunks match (or overlap with) the context in the golden dataset
"""