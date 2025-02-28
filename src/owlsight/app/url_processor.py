import asyncio
from typing import List, Optional, Union
from urllib.parse import urlparse

from owlsight.utils.logger import logger

# Check if required packages are installed
AIOHTTP_AVAILABLE = False
LXML_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    logger.warning("aiohttp package not found. Web scraping functionality will be disabled.")

try:
    import lxml.html
    LXML_AVAILABLE = True
except ImportError:
    logger.warning("lxml package not found. HTML parsing functionality will be disabled.")


def validate_url(url: str) -> bool:
    """Validate if a string is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def parse_html(html_content: Optional[str]) -> str:
    """Parse HTML content and extract text while preserving important formatting."""
    if not html_content:
        return ""
    
    if not LXML_AVAILABLE:
        logger.warning("Cannot parse HTML content: lxml package not installed.")
        return html_content  # Return raw content as fallback
        
    try:
        document = lxml.html.fromstring(html_content)
        result = []
        seen_content = set()

        # First pass: identify and mark code blocks to preserve them
        for elem in document.xpath('//pre|//code|//*[contains(@class, "highlight")]|//*[contains(@class, "code")]'):
            # Mark this element to prevent its removal
            elem.set('data-preserve', 'true')
            # Also mark all parent elements to prevent removal
            parent = elem.getparent()
            while parent is not None:
                parent.set('data-preserve', 'true')
                parent = parent.getparent()

        # Remove non-content elements
        for selector in ['//nav', '//footer', '//script', '//style', '//*[contains(@class, "ad")]']:
            for elem in document.xpath(selector):
                if elem.get('data-preserve') != 'true' and elem.getparent() is not None:
                    elem.getparent().remove(elem)

        # Find main content area
        content_selectors = [
            '//div[@role="main"]',
            '//main',
            '//article',
            '//div[contains(@class, "content")]',
            '//body'
        ]

        main_content = None
        max_content_length = 0
        
        for selector in content_selectors:
            elements = document.xpath(selector)
            for element in elements:
                # Calculate content length excluding navigation elements
                content_length = len(''.join(
                    text for text in element.xpath('.//text()[not(ancestor::nav)]')
                    if text.strip()
                ))
                if content_length > max_content_length:
                    max_content_length = content_length
                    main_content = element

        # Use an explicit check to see if a main content element was found
        if main_content is None:
            return "Could not find main content in the HTML document"

        def clean_text(text: str) -> str:
            """Clean and normalize text while preserving intentional formatting."""
            if not text:
                return ""
            # Don't modify text that appears to be code
            if any(marker in text for marker in ['```', '    ', '\t', ';', '{', '}', '[', ']']):
                return text
            # Normalize regular text
            lines = []
            for line in text.split('\n'):
                line = line.strip()
                if line:
                    lines.append(line)
            return ' '.join(lines)

        def extract_code_block(element) -> Optional[str]:
            """Extract code block content preserving exact formatting."""
            if element.tag == 'pre':
                return f'```\n{element.text_content()}\n```'
            elif element.tag == 'code':
                return f'`{element.text_content()}`'
            return None

        def process_element(element, list_level=0):
            """Process an element and its children with proper formatting."""
            if not element.tag:
                return

            # Handle code blocks
            code = extract_code_block(element)
            if code and code.strip() and code not in seen_content:
                seen_content.add(code)
                result.extend(['', code, ''])
                return

            # Handle lists
            if element.tag in ['ul', 'ol']:
                list_level += 1
            elif element.tag == 'li':
                indent = '  ' * (list_level - 1)
                result.append(f'{indent}* {clean_text(element.text_content())}')
                # Process children with current list level
                for child in element:
                    process_element(child, list_level)
            # Handle headers
            if element.tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                header_text = element.text_content().strip()
                if header_text and header_text not in seen_content:
                    seen_content.add(header_text)
                    level = int(element.tag[1])
                    result.extend(['', '#' * level + ' ' + header_text, ''])
                return

            # Handle text content
            text = element.text.strip() if element.text else ''
            if text and text not in seen_content and len(text) > 5:
                seen_content.add(text)
                if element.tag in ['p', 'div', 'section', 'article']:
                    result.extend(['', clean_text(text), ''])
                else:
                    result.append(clean_text(text))

            # Handle inline code
            elif element.tag == 'code':
                code_text = clean_text(element.text_content())
                if list_level > 0:  # Inside a list
                    result.append(f'`{code_text}`')
                else:
                    result.extend(['```', code_text, '```'])

            # Process children
            for child in element:
                process_element(child, list_level)
                # Handle tail text
                if child.tail and child.tail.strip():
                    tail_text = child.tail.strip()
                    if tail_text and tail_text not in seen_content and len(tail_text) > 5:
                        seen_content.add(tail_text)
                        result.append(clean_text(tail_text))

        # Process the main content
        process_element(main_content, 0)

        # Clean up the result
        # Remove empty lines at start and end
        while result and not result[0].strip():
            result.pop(0)
        while result and not result[-1].strip():
            result.pop()

        # Normalize multiple empty lines to single empty line
        cleaned = []
        prev_empty = False
        for line in result:
            is_empty = not line.strip()
            if not (is_empty and prev_empty):
                cleaned.append(line)
            prev_empty = is_empty

        return '\n'.join(cleaned)

    except Exception as e:
        logger.error(f"Error parsing HTML: {str(e)}")
        logger.debug(f"HTML content preview: {html_content[:200] if html_content else 'None'}")
        return ""


async def fetch_page(url: str, session = None, timeout: int = 30) -> Optional[str]:
    """
    Asynchronously fetch a webpage's content.
    
    Parameters:
    ----------
    url: str
        URL of the page to fetch
    session: aiohttp.ClientSession, optional
        Existing aiohttp client session to use
    timeout: int, optional
        Timeout in seconds for the request
    """
    if not AIOHTTP_AVAILABLE:
        logger.warning(f"Cannot fetch URL {url}: aiohttp package not installed.")
        return None
    
    if not session:
        # Create a new session if one wasn't provided
        async with aiohttp.ClientSession() as new_session:
            return await fetch_page(url, new_session, timeout)
    
    try:
        logger.info(f"Fetching {url}")
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            if response.status == 200:
                content = await response.text()
                logger.info(f"Successfully fetched {url}")
                return content
            else:
                logger.error(f"Error fetching {url}: HTTP {response.status}")
                return None
    except asyncio.TimeoutError:
        logger.error(f"Timeout fetching {url} after {timeout} seconds")
        return None
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return None


async def fetch_and_parse_urls(urls: Union[str, List[str]], max_concurrent: int = 5, timeout: int = 30) -> dict:
    """
    Async process URLs to fetch and extract markdown-formatted content.

    Parameters:
    ----------
        urls: Single URL or list of URLs to process
        max_concurrent: Maximum simultaneous requests
        timeout: Timeout in seconds for each request

    Returns:
    ----------
        Dictionary mapping URLs to their extracted content in markdown format
    """
    if isinstance(urls, str):
        urls = [urls]

    # Validate URLs first
    valid_urls = [url for url in urls if validate_url(url)]
    if not valid_urls:
        raise ValueError("No valid URLs provided")

    # Configure client session with default timeout and headers
    if not AIOHTTP_AVAILABLE:
        logger.warning("Cannot fetch URLs: aiohttp package not installed.")
        return {url: "Error: The aiohttp package is required for web scraping functionality but is not installed. Please install using 'pip install owlsight[search]'" for url in valid_urls}

    timeout_config = aiohttp.ClientTimeout(total=timeout)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    async with aiohttp.ClientSession(timeout=timeout_config, headers=headers) as session:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_wrapper(url: str) -> tuple[str, Optional[str]]:
            try:
                async with semaphore:
                    content = await fetch_page(url, session, timeout)
                    return url, content
            except Exception as e:
                logger.error(f"Failed to fetch {url}: {str(e)}")
                return url, None

        try:
            results = await asyncio.gather(*(fetch_wrapper(url) for url in valid_urls), return_exceptions=True)
            html_contents = {
                url: content
                for url, content in results
                if content is not None and not isinstance(content, Exception)
            }

            # Process HTML in executor to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            parsed_contents = {}
            for url, content in html_contents.items():
                parsed_content = await loop.run_in_executor(None, parse_html, content)
                if parsed_content:  # Only add non-empty results
                    parsed_contents[url] = parsed_content

            return parsed_contents

        except Exception as e:
            logger.error(f"Error in fetch_and_parse_urls: {str(e)}")
            return {}
