"""
AI pattern tests for Template.

Tests for common AI/LLM use case patterns: RAG, few-shot, agents, structured output.
These tests verify Template works correctly for real-world AI engineering scenarios.
"""

import pytest

from tests.template.conftest import (
    AGENT_TEMPLATE,
    FEW_SHOT_TEMPLATE,
    RAG_TEMPLATE,
    STRUCTURED_OUTPUT_TEMPLATE,
)


class TestRAGPattern:
    """Tests for Retrieval-Augmented Generation pattern."""

    def test_rag_basic(self, Template):
        """Basic RAG template with documents and question."""
        t = Template(RAG_TEMPLATE)

        documents = [
            {"source": "wiki", "content": "Paris is the capital of France."},
            {"source": "book", "content": "France is in Western Europe."},
        ]

        result = t(documents=documents, question="What is the capital of France?")

        assert "Paris is the capital of France" in result
        assert "France is in Western Europe" in result
        assert "What is the capital of France?" in result
        assert "wiki" in result
        assert "book" in result

    def test_rag_empty_documents(self, Template):
        """RAG with no documents."""
        t = Template(RAG_TEMPLATE)

        result = t(documents=[], question="What is the capital of France?")

        assert "What is the capital of France?" in result
        # Should not have any document content

    def test_rag_single_document(self, Template):
        """RAG with single document."""
        t = Template(RAG_TEMPLATE)

        documents = [{"source": "source1", "content": "Content here."}]
        result = t(documents=documents, question="Question here?")

        assert "Content here" in result
        assert "source1" in result

    def test_rag_many_documents(self, Template):
        """RAG with many documents."""
        t = Template(RAG_TEMPLATE)

        documents = [{"source": f"source{i}", "content": f"Content {i}."} for i in range(10)]
        result = t(documents=documents, question="What do you know?")

        for i in range(10):
            assert f"Content {i}" in result
            assert f"source{i}" in result

    def test_rag_unicode_content(self, Template):
        """RAG with unicode in documents."""
        t = Template(RAG_TEMPLATE)

        documents = [
            {"source": "日本", "content": "東京は日本の首都です。"},
            {"source": "Россия", "content": "Москва - столица России."},
        ]
        result = t(documents=documents, question="What are the capitals?")

        assert "東京は日本の首都です" in result
        assert "Москва - столица России" in result

    def test_rag_custom_template(self, Template):
        """Custom RAG template structure."""
        t = Template("""
<context>
{% for doc in docs %}
<document id="{{ loop.index }}">
{{ doc.text }}
</document>
{% endfor %}
</context>

<query>{{ query }}</query>
""")
        docs = [
            {"text": "First document."},
            {"text": "Second document."},
        ]
        result = t(docs=docs, query="Find info")

        assert "<context>" in result
        assert '<document id="1">' in result
        assert '<document id="2">' in result
        assert "First document" in result
        assert "<query>Find info</query>" in result


class TestFewShotPattern:
    """Tests for few-shot prompting pattern."""

    def test_few_shot_basic(self, Template):
        """Basic few-shot template."""
        t = Template(FEW_SHOT_TEMPLATE)

        examples = [
            {"input": "hello", "output": "HELLO"},
            {"input": "world", "output": "WORLD"},
        ]
        result = t(examples=examples, query="test")

        assert "Input: hello" in result
        assert "Output: HELLO" in result
        assert "Input: world" in result
        assert "Output: WORLD" in result
        assert "Input: test" in result

    def test_few_shot_zero_examples(self, Template):
        """Few-shot with zero examples (zero-shot)."""
        t = Template(FEW_SHOT_TEMPLATE)

        result = t(examples=[], query="test")

        assert "Input: test" in result
        # No examples should be present

    def test_few_shot_single_example(self, Template):
        """Few-shot with single example (one-shot)."""
        t = Template(FEW_SHOT_TEMPLATE)

        examples = [{"input": "foo", "output": "bar"}]
        result = t(examples=examples, query="baz")

        assert "foo" in result
        assert "bar" in result
        assert "baz" in result

    def test_few_shot_many_examples(self, Template):
        """Few-shot with many examples."""
        t = Template(FEW_SHOT_TEMPLATE)

        examples = [{"input": f"in{i}", "output": f"out{i}"} for i in range(20)]
        result = t(examples=examples, query="new")

        assert "in0" in result
        assert "out0" in result
        assert "in19" in result
        assert "out19" in result

    def test_few_shot_with_reasoning(self, Template):
        """Few-shot with chain-of-thought reasoning."""
        t = Template("""
{% for ex in examples %}
Input: {{ ex.input }}
Reasoning: {{ ex.reasoning }}
Output: {{ ex.output }}

{% endfor %}
Input: {{ query }}
Reasoning:
""")
        examples = [
            {"input": "2 + 3", "reasoning": "Add 2 and 3 together", "output": "5"},
        ]
        result = t(examples=examples, query="4 + 5")

        assert "Add 2 and 3" in result
        assert "4 + 5" in result


class TestAgentPattern:
    """Tests for agent/tool-use pattern."""

    def test_agent_basic(self, Template):
        """Basic agent template with tools."""
        t = Template(AGENT_TEMPLATE)

        tools = [
            {"name": "search", "description": "Search the web", "params": "query: str"},
            {"name": "calculate", "description": "Do math", "params": "expr: str"},
        ]
        result = t(tools=tools, instruction="Find the population of Tokyo")

        assert "## search" in result
        assert "Search the web" in result
        assert "## calculate" in result
        assert "Do math" in result
        assert "Find the population of Tokyo" in result

    def test_agent_no_tools(self, Template):
        """Agent with no tools available."""
        t = Template(AGENT_TEMPLATE)

        result = t(tools=[], instruction="Do something")

        assert "Do something" in result

    def test_agent_complex_tool(self, Template):
        """Agent with complex tool parameters."""
        t = Template("""
{% for tool in tools %}
Function: {{ tool.name }}
Description: {{ tool.description }}
Parameters:
{% for param in tool.params %}
  - {{ param.name }}: {{ param.type }} - {{ param.description }}
{% endfor %}

{% endfor %}

{{ instruction }}
""")
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "params": [
                    {"name": "city", "type": "string", "description": "City name"},
                    {"name": "units", "type": "string", "description": "celsius or fahrenheit"},
                ],
            }
        ]
        result = t(tools=tools, instruction="What's the weather in Paris?")

        assert "get_weather" in result
        assert "city" in result
        assert "units" in result

    def test_agent_json_tools(self, Template):
        """Agent with JSON-formatted tool definitions."""
        t = Template("""
Available tools (JSON format):
{% for tool in tools %}
{{ tool | tojson }}
{% endfor %}

Task: {{ task }}
""")
        tools = [
            {"name": "search", "params": {"query": "string"}},
        ]
        result = t(tools=tools, task="Find info")

        assert '"name"' in result or "name" in result


class TestStructuredOutputPattern:
    """Tests for structured output extraction pattern."""

    def test_structured_basic(self, Template):
        """Basic structured output template."""
        t = Template(STRUCTURED_OUTPUT_TEMPLATE)

        schema = [
            {"name": "person_name", "type": "string", "description": "The person's full name"},
            {"name": "age", "type": "integer", "description": "The person's age in years"},
        ]
        result = t(schema=schema, text="John Smith is 30 years old.")

        assert "person_name" in result
        assert "string" in result
        assert "age" in result
        assert "integer" in result
        assert "John Smith is 30 years old" in result

    def test_structured_empty_schema(self, Template):
        """Structured output with empty schema."""
        t = Template(STRUCTURED_OUTPUT_TEMPLATE)

        result = t(schema=[], text="Some text.")

        assert "Some text" in result

    def test_structured_complex_schema(self, Template):
        """Structured output with complex nested schema."""
        t = Template("""
Extract data according to this schema:
{% for field in schema %}
{{ field.name }}:
  type: {{ field.type }}
  {% if field.required %}required: true{% endif %}
  {% if field.nested %}
  properties:
  {% for prop in field.nested %}
    - {{ prop.name }}: {{ prop.type }}
  {% endfor %}
  {% endif %}
{% endfor %}

Text: {{ text }}
""")
        schema = [
            {
                "name": "address",
                "type": "object",
                "required": True,
                "nested": [
                    {"name": "street", "type": "string"},
                    {"name": "city", "type": "string"},
                ],
            }
        ]
        result = t(schema=schema, text="123 Main St, Boston")

        assert "address" in result
        assert "street" in result
        assert "city" in result


class TestSystemPromptPattern:
    """Tests for system prompt templates."""

    def test_system_prompt_basic(self, Template):
        """Basic system prompt template."""
        t = Template("""
You are {{ persona }}.

Your capabilities:
{% for cap in capabilities %}
- {{ cap }}
{% endfor %}

Rules:
{% for rule in rules %}
{{ loop.index }}. {{ rule }}
{% endfor %}
""")
        result = t(
            persona="a helpful assistant",
            capabilities=["Answer questions", "Write code"],
            rules=["Be concise", "Be accurate"],
        )

        assert "helpful assistant" in result
        assert "Answer questions" in result
        assert "1. Be concise" in result

    def test_system_prompt_conditional(self, Template):
        """System prompt with conditional sections."""
        t = Template("""
You are a {{ role }}.

{% if safety_mode %}
Safety guidelines are enabled. Always verify before executing.
{% endif %}

{% if debug_mode %}
Debug mode: Show your reasoning step by step.
{% endif %}
""")
        result = t(role="assistant", safety_mode=True, debug_mode=False)

        assert "Safety guidelines" in result
        assert "Debug mode" not in result


class TestEvaluationPattern:
    """Tests for evaluation/grading templates."""

    def test_evaluation_basic(self, Template):
        """Basic evaluation template."""
        t = Template("""
Evaluate the following response:

Question: {{ question }}
Expected Answer: {{ expected }}
Actual Response: {{ actual }}

Score (1-5):
Explanation:
""")
        result = t(question="What is 2+2?", expected="4", actual="The answer is 4")

        assert "What is 2+2?" in result
        assert "Expected Answer: 4" in result
        assert "The answer is 4" in result

    def test_evaluation_batch(self, Template):
        """Batch evaluation template."""
        t = Template("""
Evaluate these {{ cases | length }} test cases:

{% for case in cases %}
## Case {{ loop.index }}
Input: {{ case.input }}
Expected: {{ case.expected }}
Actual: {{ case.actual }}
{% if case.actual == case.expected %}Status: PASS{% else %}Status: FAIL{% endif %}

{% endfor %}
""")
        cases = [
            {"input": "2+2", "expected": "4", "actual": "4"},
            {"input": "3+3", "expected": "6", "actual": "7"},
        ]
        result = t(cases=cases)

        assert "Case 1" in result
        assert "Case 2" in result
        assert "PASS" in result
        assert "FAIL" in result


class TestMultiModalPattern:
    """Tests for multi-modal prompt templates."""

    def test_image_description_prompt(self, Template):
        """Template for image description tasks."""
        t = Template("""
Describe this image:

{% if image_url %}
[Image: {{ image_url }}]
{% endif %}

Focus on:
{% for aspect in aspects %}
- {{ aspect }}
{% endfor %}

{% if style %}
Style: {{ style }}
{% endif %}
""")
        result = t(
            image_url="https://example.com/image.jpg",
            aspects=["colors", "objects", "mood"],
            style="technical",
        )

        assert "example.com/image.jpg" in result
        assert "colors" in result
        assert "Style: technical" in result


class TestTemplateReuse:
    """Tests for template reuse patterns."""

    def test_reuse_for_batch_processing(self, Template):
        """Same template reused for batch processing."""
        t = Template("Process: {{ item }} -> Result: {{ item | upper }}")

        items = ["apple", "banana", "cherry"]
        results = [t(item=item) for item in items]

        assert "Process: apple -> Result: APPLE" in results[0]
        assert "Process: banana -> Result: BANANA" in results[1]
        assert "Process: cherry -> Result: CHERRY" in results[2]

    def test_template_library(self, Template):
        """Multiple templates used together."""
        system_template = Template("You are {{ persona }}.")
        user_template = Template("User question: {{ question }}")
        format_template = Template("Respond in {{ format }} format.")

        system = system_template(persona="an expert")
        user = user_template(question="What is AI?")
        format_instr = format_template(format="markdown")

        full_prompt = f"{system}\n\n{user}\n\n{format_instr}"

        assert "You are an expert" in full_prompt
        assert "What is AI?" in full_prompt
        assert "markdown format" in full_prompt


class TestExampleTemplates:
    """Tests for the bundled example templates in talu/examples/template/etc/.

    These tests validate that bundled example templates work correctly.
    They are marked as integration tests since they depend on files outside the package.
    """

    @pytest.fixture
    def examples_dir(self):
        """Path to the tests/template/etc directory."""
        from pathlib import Path

        # Look in tests/template/etc, next to this test file
        etc_dir = Path(__file__).parent / "etc"
        if not etc_dir.exists():
            pytest.xfail("Template fixtures directory not found")
        return etc_dir

    def test_few_shot_template(self, Template, examples_dir):
        """few_shot.j2 template works correctly."""
        t = Template.from_file(str(examples_dir / "few_shot.j2"))

        result = t(
            examples=[
                {"text": "I love this!", "sentiment": "positive"},
                {"text": "This is terrible.", "sentiment": "negative"},
            ],
            query="Best purchase ever!",
        )

        assert "I love this!" in result
        assert "positive" in result
        assert "Best purchase ever!" in result

    def test_rag_template(self, Template, examples_dir):
        """rag.j2 template works correctly."""
        t = Template.from_file(str(examples_dir / "rag.j2"))

        result = t(
            documents=[
                {"source": "FAQ", "content": "Returns accepted within 30 days."},
                {"source": "Policy", "content": "Refunds processed in 5-7 days."},
            ],
            question="How long do refunds take?",
        )

        assert "FAQ" in result
        assert "Returns accepted" in result
        assert "How long do refunds take?" in result

    def test_tool_use_template(self, Template, examples_dir):
        """tool_use.j2 template works correctly."""
        t = Template.from_file(str(examples_dir / "tool_use.j2"))

        result = t(
            tools=[
                {"name": "search", "description": "Search the web", "parameters": ["query"]},
            ],
            user_request="Find the weather in Tokyo",
        )

        assert "search" in result
        assert "Search the web" in result
        assert "Find the weather in Tokyo" in result

    def test_chain_of_thought_template(self, Template, examples_dir):
        """chain_of_thought.j2 template works correctly."""
        t = Template.from_file(str(examples_dir / "chain_of_thought.j2"))

        result = t(
            problem="What is 15% of 80?", hints=["Convert percentage to decimal", "Multiply"]
        )

        assert "What is 15% of 80?" in result
        assert "Convert percentage" in result
        assert "step by step" in result

    def test_summarize_template(self, Template, examples_dir):
        """summarize.j2 template works correctly."""
        t = Template.from_file(str(examples_dir / "summarize.j2"))

        result = t(
            document_type="email", content="Hi team, just wanted to follow up on our meeting..."
        )

        assert "email" in result
        assert "follow up" in result

    def test_extract_template(self, Template, examples_dir):
        """extract.j2 template works correctly."""
        t = Template.from_file(str(examples_dir / "extract.j2"))

        result = t(
            fields=["name", "email", "phone"],
            text="Contact John at john@example.com or call 555-1234",
        )

        assert "name" in result
        assert "email" in result
        assert "phone" in result
        assert "John" in result
