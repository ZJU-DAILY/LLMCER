
import os

# Base Directory (Project Root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv(path):
    """
    Minimal .env loader (no external dependency). Reads KEY=VALUE lines and
    populates os.environ for any key not already set. The .env file is
    git-ignored so the API key never gets committed.
    """
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


_load_dotenv(os.path.join(BASE_DIR, ".env"))

# OpenAI API Key (read from environment / .env -- never hard-coded here).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_api_key_here")

# Optional custom API base URL. Set this when the key belongs to an
# OpenAI-COMPATIBLE gateway / proxy (e.g. a relay, Azure, or a provider like
# PPIO) rather than api.openai.com. Leave empty to use the official endpoint.
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip() or None

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

# In-context clustering record-set knobs (paper Section 4.2 / 6.3 optima).
SET_SIZE = 9       # S_s : records per in-context clustering call
SET_DIVERSITY = 4  # S_d : distinct entities targeted per record set

# Similarity thresholds.
# These are FIXED, validation-tuned values (paper Section 5.1: b_t is chosen by
# maximising F1 on a validation set over 0.05..0.95). They are NOT derived from
# the per-dataset mean/std of the similarity matrix -- doing so made the merge
# band (merge_threshold, block_threshold) collapse to empty on datasets whose
# similarity distribution differs, so no merging happened. Override per dataset
# via env vars if a different validation optimum is found.
import os as _os
BLOCK_THRESHOLD = float(_os.getenv("BLOCK_THRESHOLD", "0.70"))   # b_t for LSH blocking
SEPARATION_THRESHOLD = float(_os.getenv("SEPARATION_THRESHOLD", "0.70"))
MERGE_THRESHOLD = float(_os.getenv("MERGE_THRESHOLD", "0.50"))

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
