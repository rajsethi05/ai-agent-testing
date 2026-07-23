import os
import time

from deepeval.models import GPTModel
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from dotenv import load_dotenv

from config import ROOT_DIR

load_dotenv()
model = GPTModel(model="gpt-4.1-mini")

transcript_dir = os.path.join(ROOT_DIR, "agents/rag_youtube_chatbot/transcripts")

transcript_files = os.listdir(transcript_dir)

for transcript_file in transcript_files:
    synthesizer = Synthesizer(model=model, max_concurrent=10)
    synthesizer.generate_goldens_from_docs(document_paths=[f"{transcript_dir}/{transcript_file}"],
                                           include_expected_output=True, max_goldens_per_context=2,
                                           context_construction_config=ContextConstructionConfig(
                                               # Chunk size here is the token and not words. 4words=1token
                                               chunk_size=150, chunk_overlap=30, ), )
    synthesizer.save_as(file_type="json", directory=f"{ROOT_DIR}/framework/golden_dataset/datasets",
                        file_name=transcript_file.replace(".txt", ""))
    print("Done: ", transcript_file)
    print("sleeping...")
    time.sleep(60)
