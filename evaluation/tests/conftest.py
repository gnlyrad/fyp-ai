import os
import sys
from pathlib import Path

# Tests use fake graphs and fake models; never send their traces to LangSmith, even when the
# repository .env enables tracing for real runs (dotenv does not override variables already set).
for _name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING"):
    os.environ[_name] = "false"

# Price lookups are exercised with injected catalogs, never the real download.
os.environ["EVALUATION_PRICE_CATALOG_URL"] = "off"

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))
