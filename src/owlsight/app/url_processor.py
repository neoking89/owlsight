import asyncio
from typing import List, Optional, Union
from urllib.parse import urlparse
import aiohttp
import lxml.html

from owlsight.utils.logger import logger


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

        # Remove non-content elements, preserving marked elements
        for elem in document.xpath('//script|//style|//link|//meta|//noscript|//iframe'):
            if elem.getparent() is not None and not elem.get('data-preserve'):
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

        if not main_content:
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
            try:
                # Get the raw HTML to preserve exact formatting
                raw_html = lxml.html.tostring(element, encoding='unicode')
                # If it's a pre element, preserve all whitespace exactly
                if element.tag == 'pre':
                    code = element.text_content()
                    if code.strip():
                        return code
                # For other elements, check if it contains code-like content
                code = element.text_content()
                if code.strip() and any(marker in code for marker in [';', '{', '}', '(', ')', '[', ']', '=', 'def ', 'class ']):
                    return code
            except Exception:
                pass
            return None

        def process_element(element, level=0):
            """Process an element and its children with proper formatting."""
            if not element.tag:
                return

            # Handle code blocks first
            if (element.tag in ['pre', 'code'] or 
                any(cls in (element.get('class') or '').lower() for cls in ['highlight', 'code', 'syntax', 'source'])):
                code = extract_code_block(element)
                if code and code.strip() and code not in seen_content:
                    seen_content.add(code)
                    # For single-line code, use inline code format
                    if '\n' not in code and len(code) < 100:
                        result.extend(['', f'`{code.strip()}`', ''])
                    else:
                        # For multi-line code, preserve exact formatting
                        result.extend(['', '```', code, '```', ''])
                return

            # Handle headers
            if element.tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                header_text = element.text_content().strip()
                if header_text and header_text not in seen_content:
                    seen_content.add(header_text)
                    level = int(element.tag[1])
                    result.extend(['', '#' * level + ' ' + header_text, ''])
                return

            # Handle lists
            if element.tag == 'li':
                list_text = element.text_content().strip()
                if list_text and list_text not in seen_content:
                    seen_content.add(list_text)
                    result.append('* ' + clean_text(list_text))
                return

            # Handle text content
            text = element.text.strip() if element.text else ''
            if text and text not in seen_content and len(text) > 5:
                seen_content.add(text)
                if element.tag in ['p', 'div', 'section', 'article']:
                    result.extend(['', clean_text(text), ''])
                else:
                    result.append(clean_text(text))

            # Process children
            for child in element:
                process_element(child, level + 1)
                # Handle tail text
                if child.tail and child.tail.strip():
                    tail_text = child.tail.strip()
                    if tail_text and tail_text not in seen_content and len(tail_text) > 5:
                        seen_content.add(tail_text)
                        result.append(clean_text(tail_text))

        # Process the main content
        process_element(main_content)

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


async def fetch_page(url: str, session: aiohttp.ClientSession, timeout: int = 30) -> Optional[str]:
    """Asynchronously fetch a webpage's content."""
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
    """Async process URLs to fetch and extract markdown-formatted content.

    Parameters:
        urls: Single URL or list of URLs to process
        max_concurrent: Maximum simultaneous requests
        timeout: Timeout in seconds for each request

    Returns:
        Dictionary mapping URLs to their extracted content in markdown format
    """
    if isinstance(urls, str):
        urls = [urls]

    # Validate URLs first
    valid_urls = [url for url in urls if validate_url(url)]
    if not valid_urls:
        raise ValueError("No valid URLs provided")

    # Configure client session with default timeout and headers
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
