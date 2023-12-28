import logging
import sys
from dotenv import load_dotenv
from llama_index.llms import OpenAI
import os.path
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    load_index_from_storage,
)

load_dotenv()

# Get the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_model = os.getenv("OPENAI_API_MODEL")
persist_dir = os.getenv("PERSIST_DIR")

# Create a ServiceContext instance with the OpenAI API key
service_context = ServiceContext.from_defaults(
    llm=OpenAI(api_key=openai_api_key, model=openai_api_model)
)

# check if storage already exists
if not os.path.exists(persist_dir):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    # store it for later
    index.storage_context.persist(persist_dir=persist_dir)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)

# either way we can now query the index
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up? 日本語で答えてください。")
print(response)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
