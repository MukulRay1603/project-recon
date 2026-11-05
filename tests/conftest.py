import os
import sys
import tempfile

# Make `import src...` work when pytest is run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Keep the on-disk API cache out of the repo during tests.
os.environ.setdefault("RECON_CACHE_DIR", tempfile.mkdtemp(prefix="recon-test-cache-"))
