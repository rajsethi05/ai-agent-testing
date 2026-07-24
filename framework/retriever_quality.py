from agents.rag_youtube_chatbot.yt_chatbot import YTChatbot

yt = YTChatbot("HAoKJT3af7Y")
# retriever = yt.create_retriever()
res= yt.get_answer("How does Deep Eval automate admin-side comparison of cloud, Gemini, OpenAI, and local LLMs?")
ret = yt.retriever.invoke("How does Deep Eval automate admin-side comparison of cloud, Gemini, OpenAI, and local LLMs?")
# results = retriever.invoke("How does shrinkflation reduce product size while maintaining or increasing retail prices?")
# result = "--".join(res.page_content for res in results)
print(res)
# print(results)


"""
Exactly. The test flow is:

Take input from golden dataset
Pass it to your RAG retriever
Get the retrieved chunks
Verify those retrieved chunks match (or overlap with) the context in the golden dataset
"""