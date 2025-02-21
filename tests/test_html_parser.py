import pytest
from owlsight.app.url_processor import parse_html

def normalize_text(text):
    """Normalize text for comparison by removing extra whitespace."""
    if not text:
        return ""
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(line for line in lines if line)

class TestHTMLParser:
    def test_empty_input(self):
        assert parse_html(None) == ""
        assert parse_html("") == ""
        assert parse_html("   ") == ""

    def test_simple_text(self):
        html = "<p>Hello world</p>"
        result = normalize_text(parse_html(html))
        assert "Hello world" in result

    def test_headers(self):
        html = """
        <div>
            <h1>Main Title</h1>
            <h2>Subtitle</h2>
            <h3>Section 1</h3>
        </div>
        """
        result = normalize_text(parse_html(html))
        assert "# Main Title" in result
        assert "## Subtitle" in result
        assert "### Section 1" in result

    def test_code_blocks(self):
        html = """
        <pre><code>def example():
    return "Hello"</code></pre>
        """
        result = parse_html(html)
        assert "```" in result
        assert "def example():" in result
        assert "    return \"Hello\"" in result

    def test_inline_code(self):
        html = "<p>Use <code>print()</code> function</p>"
        result = parse_html(html)
        assert "`print()`" in result

    def test_nested_lists(self):
        html = """
        <ul>
            <li>First level
                <ul>
                    <li>Second level A</li>
                    <li>Second level B</li>
                </ul>
            </li>
            <li>Another first level</li>
        </ul>
        """
        result = parse_html(html)
        assert "* First level" in result
        assert "  * Second level A" in result
        assert "  * Second level B" in result
        assert "* Another first level" in result

    def test_complex_document(self):
        html = """
        <article class="markdown-body">
            <h1>Project Documentation</h1>
            <p>Welcome to the documentation.</p>
            
            <h2>Installation</h2>
            <pre><code>pip install mypackage</code></pre>
            
            <h2>Usage</h2>
            <p>Here's a simple example:</p>
            <pre><code>
            from mypackage import MyClass
            
            obj = MyClass()
            obj.do_something()
            </code></pre>
            
            <h3>Configuration</h3>
            <p>Configure using a dictionary:</p>
            <pre><code>
            config = {
                "host": "localhost",
                "port": 8080
            }
            </code></pre>
        </article>
        """
        result = parse_html(html)
        expected_elements = [
            "# Project Documentation",
            "Welcome to the documentation",
            "## Installation",
            "pip install mypackage",
            "## Usage",
            "Here's a simple example",
            "from mypackage import MyClass",
            "obj = MyClass()",
            "obj.do_something()",
            "### Configuration",
            "Configure using a dictionary",
            '"host": "localhost"',
            '"port": 8080'
        ]
        for element in expected_elements:
            assert element in result

    def test_mixed_content(self):
        html = """
        <div class="content">
            <h1>API Guide</h1>
            <p>This guide explains the API.</p>
            <pre><code>import requests
response = requests.get('https://api.example.com')</code></pre>
            <ul>
                <li>Simple to use</li>
                <li>Well documented
                    <ul>
                        <li>Full examples</li>
                        <li>API reference</li>
                    </ul>
                </li>
            </ul>
            <p>For more information, see the docs.</p>
        </div>
        """
        result = parse_html(html)
        assert "# API Guide" in result
        assert "This guide explains the API" in result
        assert "import requests" in result
        assert "* Simple to use" in result
        assert "* Well documented" in result
        assert "  * Full examples" in result
        assert "  * API reference" in result

    def test_preserve_code_indentation(self):
        html = """
        <pre><code>
def example():
    try:
        result = process()
        if result:
            return True
    except Exception as e:
        logger.error(str(e))
        return False</code></pre>
        """
        result = parse_html(html)
        assert "def example():" in result
        assert "    try:" in result
        assert "        result = process()" in result
        assert "    except Exception as e:" in result

    def test_remove_unwanted_elements(self):
        html = """
        <div>
            <nav>Skip this navigation</nav>
            <main>
                <h1>Main Content</h1>
                <p>Keep this content</p>
            </main>
            <footer>Skip this footer</footer>
            <div class="ad">Skip this ad</div>
        </div>
        """
        result = normalize_text(parse_html(html))
        assert "Skip this navigation" not in result
        assert "Skip this footer" not in result
        assert "Skip this ad" not in result
        assert "# Main Content" in result
        assert "Keep this content" in result

    def test_markdown_formatting(self):
        html = """
        <article>
            <h1>Title</h1>
            <p>Normal paragraph.</p>
            <pre><code>code block</code></pre>
            <ul>
                <li>List item 1</li>
                <li>List item 2</li>
            </ul>
            <p>Another <code>inline code</code> example.</p>
        </article>
        """
        result = normalize_text(parse_html(html))
        assert "# Title" in result
        assert "Normal paragraph" in result
        assert "```\ncode block\n```" in result
        assert "* List item 1" in result
        assert "* List item 2" in result
        assert "`inline code`" in result

    def test_complex_code_blocks(self):
        html = """
        <div class="example">
            <pre class="syntax-highlight"><code>
class Example:
    def __init__(self, value: int):
        self.value = value
    
    def process(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "processed": True
        }
            </code></pre>
        </div>
        """
        result = parse_html(html)
        assert "class Example:" in result
        assert "    def __init__(self, value: int):" in result
        assert "        self.value = value" in result
        assert "    def process(self) -> Dict[str, Any]:" in result
        assert '            "value": self.value,' in result

    def test_nested_content(self):
        html = """
        <div class="container">
            <h2>Features</h2>
            <div class="feature">
                <h3>Easy Integration</h3>
                <p>Integrate with any system using our API:</p>
                <pre><code>client.connect()</code></pre>
                <ul>
                    <li>Simple setup
                        <ul>
                            <li>No configuration needed</li>
                            <li>Works out of the box</li>
                        </ul>
                    </li>
                    <li>Extensive documentation</li>
                </ul>
            </div>
        </div>
        """
        result = parse_html(html)
        assert "## Features" in result
        assert "### Easy Integration" in result
        assert "Integrate with any system using our API:" in result
        assert "`client.connect()`" in result
        assert "* Simple setup" in result
        assert "  * No configuration needed" in result
        assert "  * Works out of the box" in result
        assert "* Extensive documentation" in result
