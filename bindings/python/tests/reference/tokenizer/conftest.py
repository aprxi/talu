"""Reference tokenizer fixtures.

Pytest 9 removed support for ``pytest_plugins`` declarations in non-top-level
conftest files. Re-export tokenizer fixtures directly for this subtree.
"""

from tests.tokenizer.conftest import *  # noqa: F401,F403

