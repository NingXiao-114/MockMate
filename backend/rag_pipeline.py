import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")

GRADE_MODEL = os.getenv("GRADE_MODEL")
GRADE_API_KEY = os.getenv("GRADE_API_KEY")
GRADE_BASE_URL = os.getenv("GRADE_BASE_URL")




_grader_model = None
_router_model = None

def _get_grader_model():
    global _grader_model
    if not API_KEY or not GRADE_MODEL:
        return None
    if _grader_model is None:
        _grader_model = init_chat_model(
            model=GRADE_MODEL,
            model_provider="openai",
            api_key=GRADE_API_KEY,
            base_url=GRADE_BASE_URL,
            temperature=0,
            stream_usage=True,
        )
    return _grader_model

def _get_router_model():
    global _router_model
    if not API_KEY or not MODEL:
        return None
    if _router_model is None:
        _router_model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
            stream_usage=True,
        )
    return _router_model

