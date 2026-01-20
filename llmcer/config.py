
import os

# Base Directory (Project Root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_api_key_here")

# Model Configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Embedding Model
# Check if local model exists in project root
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "all-MiniLM-L6-v2")
if os.path.exists(LOCAL_MODEL_PATH):
    EMBEDDING_MODEL = LOCAL_MODEL_PATH
else:
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Allow override via env var
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_PATH", EMBEDDING_MODEL)

# Paths
DATASET_PATH = os.getenv("DATASET_PATH", os.path.join(BASE_DIR, "datasets", "demo_dataset", "demo_er_dataset.xlsx"))
GROUND_TRUTH_PATH = os.getenv("GROUND_TRUTH_PATH", os.path.join(BASE_DIR, "datasets", "demo_dataset", "demo_er_ground_truth.txt"))

# Clustering
MAX_K = 5
LSH_HASH_SIZE = 15
LSH_INPUT_DIM = 384
LSH_NUM_HASHTABLES = 8

# Prompts
PRE_PROMPT_CLASSIFY = (
    "Please classify the following records into a two-dimensional list. Each element of the array "
    "should be a group, containing the record IDs of that group (e.g., 1, 2, 3, etc.). Ensure that "
    "each record ID is classified exactly once and appear once in the 2D array, without any "
    "duplication or omission.The output should be a two-dimensional list with no additional information!\n"
)

PRE_PROMPT_MERGE = (
    "Do the records in the following clusters refer to the same entity? i.e., given that the records in each "
    "cluster refer to the same entity, can these clusters or parts of these clusters be merged? If they all "
    "point to one entity, answer 'Yes' And returns a two-dimensional array, each dimension of the array is "
    "the cluster id, indicating which clusters can be clustered together, otherwise just answer 'No' with "
    "no reason.You only need to tell me yes or no!!!\n"
)
