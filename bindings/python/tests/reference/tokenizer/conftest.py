"""
Reference tokenizer test fixtures.

Expose shared tokenizer fixtures to the reference tokenizer subtree without
registering them globally for the full test suite.
"""

pytest_plugins = ["tests.tokenizer.conftest"]
