import os
from pathlib import Path
import sys

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi

from agents.logging_config import get_logger

parent_dir = os.path.dirname(__file__)

class Config:
    transcripts_path = os.path.join(parent_dir, "transcripts")
    vector_store_path = os.path.join(parent_dir, "vector_stores")

    ''' ************* Configs ******************* '''
    embedding_model_name = "text-embedding-3-small"
    llm_model_name = "gpt-5-mini"
    ''' *************************************** '''
    prompt = ChatPromptTemplate([("system", """
            You are a helpful assistant. Answer questions about the video. Use the provided transcript context.
            {context}"""), ("human", "{question}")])
'''
    prompt = ChatPromptTemplate([("system", """
    You are a helpful assistant. Your job is to provide the answer to the user's question with explanation.
    Answer the question only on the basis of information provided in the context and do not add any information from your side.
    The answer should look like you are providing it. Do not talk from the context's perspective.
    If the context is insufficient, just say I don't know.
    {context}"""), ("human", "{question}")])
'''

''' ********** vars **************** '''
load_dotenv()
embedding = OpenAIEmbeddings(model=Config.embedding_model_name)
llm = ChatOpenAI(model=Config.llm_model_name)
logger = get_logger("chatbot")
parser = StrOutputParser()
''' ********************************* '''

class YTChatbot:

    def __init__(self, video_id):
        self.retriever = None
        self.video_id = video_id
        self.transcript_file_name = video_id + ".txt"
        self.transcript = None

    @staticmethod
    def format_context(context_list):
        return " ".join(context.page_content for context in context_list)

    def get_transcript(self):
        os.makedirs(Config.transcripts_path, exist_ok=True)
        saved_transcripts = os.listdir(Config.transcripts_path)
        if self.transcript_file_name in saved_transcripts:
            logger.info(f"Found transcript file: {self.transcript_file_name}")
            transcript = Path(os.path.join(Config.transcripts_path, self.transcript_file_name)).read_text()
            logger.info("Transcript file loaded")
        else:
            logger.info(f"Transcript file not found. Trying to download...")
            yt_api = YouTubeTranscriptApi()
            try:
                transcript_raw = yt_api.fetch(self.video_id)
            except Exception as e:
                logger.error(f"Failed to fetch transcript from YouTube: {e.ERROR_MESSAGE}")
                return None

            transcript = " ".join(txt.text for txt in transcript_raw)
            if transcript == "":
                logger.error(f"Unable to fetch transcript for {self.video_id}")
                sys.exit()
            logger.info(f"Transcript fetched for {self.video_id}")
            with open(os.path.join(Config.transcripts_path, self.transcript_file_name), "w") as text_file:
                text_file.write(transcript)
                logger.info(f"Transcript file saved to {Config.transcripts_path.rstrip(os.sep)}")
        self.transcript = transcript
        return self

    def clean_transcript(self):
        prompt = f"""You are given a YouTube video transcript. 
        Remove all advertisement, sponsor, and promotional segments from the transcript.
        Keep all other content exactly as is — do not paraphrase, summarize, or alter the remaining text.
        Return only the cleaned transcript.

        Transcript:
        {self.transcript}"""

        response = llm.invoke(prompt)
        return response.content

    @staticmethod
    def split_text(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
        logger.info("Text splitting done.")
        return text_splitter.split_text(text)

    def get_or_create_vector_store(self):
        logger.info("Checking vector store.")
        client = chromadb.PersistentClient(path=Config.vector_store_path)
        vector_collections = [collection.name for collection in client.list_collections()]

        if self.video_id in vector_collections:
            logger.info("Existing vector store found.")
            return Chroma(collection_name=self.video_id, embedding_function=embedding,
                          persist_directory=Config.vector_store_path)
        else:
            logger.info("No stored vector found. Generating and storing vectors")

            # Step1 Get transcript
            video_text = self.get_transcript().clean_transcript()

            if video_text is None:
                return None
            # Step 2 Split Text
            splitted_text = self.split_text(video_text)
            return Chroma.from_texts(texts=splitted_text, embedding=embedding, collection_name=self.video_id,
                                     persist_directory=Config.vector_store_path)

    def create_retriever(self):
        vector = self.get_or_create_vector_store()
        if vector:
            self.retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        return self.retriever

    def get_answer(self, question):
        retriever_chain = RunnableParallel(
            {"context": self.retriever | RunnableLambda(self.format_context), "question": RunnablePassthrough()})

        # final_chain = retriever_chain | Config.prompt
        final_chain = retriever_chain | Config.prompt | llm | parser
        result = final_chain.invoke(question)
        return result

logger.info("*********** New Chat started *********************")

# vectors = get_or_create_vector_store()

# Step 4 Perform search
# retriever = vectors.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# question = "How many kinds of procrastinators are there ?"


# *************************** Without Using Chains *****************************

# context = format_context(retriever.invoke(question))
# logger.info("Context retrieved")
#
#
# final_prompt = prompt.invoke({"context": context, "question": question})
# answer = llm.invoke(final_prompt)
# logger.info("LLM response received")
# print(answer.content)

# *************************** Using Chains *****************************
# question = "Who are the founders of MM0 ?"
# yt= YTChatbot("Sr1STQP0cds")
# retriever = yt.get_or_create_vector_store().as_retriever(search_type="similarity", search_kwargs={"k": 2})
#
# retriever_chain = RunnableParallel({
#     "context": retriever | RunnableLambda(yt.format_context),
#     "question": RunnablePassthrough()
# })
#
# final_chain = retriever_chain | Config.prompt | llm | parser
# result  = final_chain.invoke(question)
# print(result)
