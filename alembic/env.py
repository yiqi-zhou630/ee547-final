# from app.db.base import Base
# from app.core.config import settings
# from sqlalchemy import engine_from_config, pool
#
# config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
# target_metadata = Base.metadata

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # ee547-final/
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
from app.db.base import Base
from app.core.config import settings
