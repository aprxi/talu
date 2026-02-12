"""Tests for the parse_functions template filter.

The parse_functions filter extracts function documentation from Python source code,
enabling templates to describe Python functions without Python-side introspection.
"""

import talu


class TestParseFunctionsFilter:
    """Test the parse_functions filter for extracting function documentation."""

    def test_simple_function(self):
        """Parse a simple function with docstring."""
        template = talu.PromptTemplate(
            "{% for fn in source | parse_functions %}{{ fn.name }}: {{ fn.description }}{% endfor %}"
        )
        source = '''
def hello(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"
'''
        result = template(source=source)
        assert result == "hello: Say hello to someone."

    def test_function_with_parameters(self):
        """Parse function parameters with types."""
        template = talu.PromptTemplate(
            """{% for fn in source | parse_functions %}{{ fn.name }}({% for p in fn.parameters %}{{ p.name }}: {{ p.type }}{% if not loop.last %}, {% endif %}{% endfor %}){% endfor %}"""
        )
        source = '''
def search(query: str, limit: int = 10) -> list:
    """Search for something."""
    pass
'''
        result = template(source=source)
        assert result == "search(query: string, limit: integer)"

    def test_parameter_required_status(self):
        """Parameters with defaults are not required."""
        template = talu.PromptTemplate(
            """{% for fn in source | parse_functions %}{% for p in fn.parameters %}{{ p.name }}:{{ p.required }}{% if not loop.last %},{% endif %}{% endfor %}{% endfor %}"""
        )
        source = '''
def func(required_param: str, optional_param: int = 10):
    """Test function."""
    pass
'''
        result = template(source=source)
        assert result == "required_param:True,optional_param:False"

    def test_parameter_default_values(self):
        """Default values are captured as strings."""
        template = talu.PromptTemplate(
            """{% for fn in source | parse_functions %}{% for p in fn.parameters %}{{ p.name }}={{ p.default | default('none') }}{% if not loop.last %},{% endif %}{% endfor %}{% endfor %}"""
        )
        source = '''
def func(a: str, b: int = 10, c: str = "hello"):
    """Test function."""
    pass
'''
        result = template(source=source)
        assert result == 'a=none,b=10,c="hello"'

    def test_google_style_docstring(self):
        """Parse Google-style docstring with Args section."""
        template = talu.PromptTemplate("""{% for fn in source | parse_functions %}{{ fn.name }}: {{ fn.description }}
{% for p in fn.parameters %}- {{ p.name }}: {{ p.description }}
{% endfor %}{% endfor %}""")
        source = '''
def search(query: str, limit: int = 10) -> list:
    """Search the web for information.

    Args:
        query: The search query
        limit: Maximum results to return
    """
    pass
'''
        result = template(source=source)
        expected = """search: Search the web for information.
- query: The search query
- limit: Maximum results to return
"""
        assert result == expected

    def test_multiple_functions(self):
        """Parse multiple functions from source."""
        template = talu.PromptTemplate(
            "{% for fn in source | parse_functions %}{{ fn.name }}{% if not loop.last %}, {% endif %}{% endfor %}"
        )
        source = '''
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
'''
        result = template(source=source)
        assert result == "add, multiply"

    def test_async_function(self):
        """Async functions are marked as async."""
        template = talu.PromptTemplate(
            "{% for fn in source | parse_functions %}{{ fn.name }}: async={{ fn.async }}{% endfor %}"
        )
        source = '''
async def fetch(url: str) -> str:
    """Fetch a URL."""
    pass
'''
        result = template(source=source)
        assert result == "fetch: async=True"

    def test_optional_type(self):
        """Optional[T] types are marked as not required."""
        template = talu.PromptTemplate(
            "{% for fn in source | parse_functions %}{% for p in fn.parameters %}{{ p.name }}:{{ p.type }}:{{ p.required }}{% if not loop.last %},{% endif %}{% endfor %}{% endfor %}"
        )
        source = '''
def find(name: str, default: Optional[str] = None) -> str:
    """Find something."""
    pass
'''
        result = template(source=source)
        assert result == "name:string:True,default:string:False"

    def test_no_type_hints(self):
        """Functions without type hints get 'any' type."""
        template = talu.PromptTemplate(
            "{% for fn in source | parse_functions %}{% for p in fn.parameters %}{{ p.name }}:{{ p.type }}{% endfor %}{% endfor %}"
        )
        source = '''
def process(data):
    """Process some data."""
    pass
'''
        result = template(source=source)
        assert result == "data:any"

    def test_return_type(self):
        """Return type is extracted."""
        template = talu.PromptTemplate(
            "{% for fn in source | parse_functions %}{{ fn.name }} -> {{ fn.return_type }}{% endfor %}"
        )
        source = '''
def get_items() -> list:
    """Get items."""
    pass
'''
        result = template(source=source)
        assert result == "get_items -> array"

    def test_decorated_function(self):
        """Decorators are skipped, function still parsed."""
        template = talu.PromptTemplate(
            "{% for fn in source | parse_functions %}{{ fn.name }}{% endfor %}"
        )
        source = '''
@decorator
@another_decorator
def decorated_func(x: int) -> int:
    """A decorated function."""
    return x
'''
        result = template(source=source)
        assert result == "decorated_func"

    def test_type_mapping(self):
        """Python types are mapped to JSON Schema-like types."""
        template = talu.PromptTemplate(
            "{% for fn in source | parse_functions %}{% for p in fn.parameters %}{{ p.type }}{% if not loop.last %},{% endif %}{% endfor %}{% endfor %}"
        )
        source = '''
def func(a: str, b: int, c: float, d: bool, e: list, f: dict) -> None:
    """Test type mapping."""
    pass
'''
        result = template(source=source)
        assert result == "string,integer,number,boolean,array,object"

    def test_empty_source(self):
        """Empty source returns empty array."""
        template = talu.PromptTemplate(
            "{% for fn in source | parse_functions %}{{ fn.name }}{% endfor %}"
        )
        result = template(source="")
        assert result == ""

    def test_no_functions(self):
        """Source without functions returns empty array."""
        template = talu.PromptTemplate(
            "{% for fn in source | parse_functions %}{{ fn.name }}{% endfor %}"
        )
        source = """
x = 1
y = 2
"""
        result = template(source=source)
        assert result == ""

    def test_tool_description_example(self):
        """Real-world example: generate tool descriptions for LLM."""
        template = talu.PromptTemplate("""Available tools:
{% for fn in tools_source | parse_functions %}
## {{ fn.name }}
{{ fn.description }}
{% if fn.parameters %}
Parameters:
{% for p in fn.parameters %}- {{ p.name }} ({{ p.type }}{% if not p.required %}, optional{% endif %}): {{ p.description | default("No description") }}{% if p.default %} Default: {{ p.default }}{% endif %}

{% endfor %}{% endif %}{% endfor %}""")

        tools_source = '''
def search(query: str, limit: int = 10) -> list:
    """Search the web for information.

    Args:
        query: The search query to execute
        limit: Maximum number of results to return
    """
    pass

def calculate(expression: str) -> float:
    """Evaluate a mathematical expression.

    Args:
        expression: The math expression to evaluate
    """
    pass
'''

        result = template(tools_source=tools_source)

        # Verify key parts are present
        assert "## search" in result
        assert "Search the web for information." in result
        assert "- query (string): The search query to execute" in result
        assert "- limit (integer, optional): Maximum number of results to return" in result
        assert "Default: 10" in result

        assert "## calculate" in result
        assert "Evaluate a mathematical expression." in result
        assert "- expression (string): The math expression to evaluate" in result
