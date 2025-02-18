#!/usr/bin/env python3

import asyncio
from typing import List, Optional, Union
import html5lib
from urllib.parse import urlparse
import aiohttp

from owlsight.utils.logger import logger


def validate_url(url: str) -> bool:
    """Validate if a string is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def parse_html(html_content: Optional[str]) -> str:
    """Parse HTML content and extract text with hyperlinks in markdown format."""
    if not html_content:
        return ""

    try:
        document = html5lib.parse(html_content)
        result = []
        seen_texts = set()  # To avoid duplicates

        def should_skip_element(elem) -> bool:
            """Check if the element should be skipped."""
            # Skip script and style tags
            if elem.tag in ["{http://www.w3.org/1999/xhtml}script", "{http://www.w3.org/1999/xhtml}style"]:
                return True
            # Skip empty elements or elements with only whitespace
            if not any(text.strip() for text in elem.itertext()):
                return True
            return False

        def process_element(elem, depth=0):
            """Process an element and its children recursively."""
            if should_skip_element(elem):
                return

            # Handle text content
            if hasattr(elem, "text") and elem.text:
                text = elem.text.strip()
                if text and text not in seen_texts:
                    # Check if this is an anchor tag
                    if elem.tag == "{http://www.w3.org/1999/xhtml}a":
                        href = None
                        for attr, value in elem.items():
                            if attr.endswith("href"):
                                href = value
                                break
                        if href and not href.startswith(("#", "javascript:")):
                            # Format as markdown link
                            link_text = f"[{text}]({href})"
                            result.append("  " * depth + link_text)
                            seen_texts.add(text)
                    else:
                        result.append("  " * depth + text)
                        seen_texts.add(text)

            # Process children
            for child in elem:
                process_element(child, depth + 1)

            # Handle tail text
            if hasattr(elem, "tail") and elem.tail:
                tail = elem.tail.strip()
                if tail and tail not in seen_texts:
                    result.append("  " * depth + tail)
                    seen_texts.add(tail)

        # Start processing from the body tag
        body = document.find(".//{http://www.w3.org/1999/xhtml}body")
        if body is not None:
            process_element(body)
        else:
            # Fallback to processing the entire document
            process_element(document)

        # Filter out common unwanted patterns
        filtered_result = []
        for line in result:
            # Skip lines that are likely to be noise
            if any(
                pattern in line.lower()
                for pattern in ["var ", "function()", ".js", ".css", "google-analytics", "disqus", "{", "}"]
            ):
                continue
            filtered_result.append(line)

        return "\n".join(filtered_result)
    except Exception as e:
        logger.error(f"Error parsing HTML: {str(e)}")
        return ""


async def fetch_page(url: str, session: aiohttp.ClientSession, timeout: int = 30) -> Optional[str]:
    """Asynchronously fetch a webpage's content."""
    try:
        logger.info(f"Fetching {url}")
        # Add timeout to prevent hanging
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
    timeout_config = aiohttp.ClientTimeout(total=timeout)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    async with aiohttp.ClientSession(timeout=timeout_config, headers=headers) as session:
        # Fetch pages with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_wrapper(url: str) -> tuple[str, Optional[str]]:
            try:
                async with semaphore:
                    content = await fetch_page(url, session, timeout)
                    return url, content
            except Exception as e:
                logger.error(f"Failed to fetch {url}: {str(e)}")
                return url, None

        # Run all fetches concurrently with proper exception handling
        try:
            results = await asyncio.gather(*(fetch_wrapper(url) for url in valid_urls), return_exceptions=True)

            # Filter out exceptions and create a dictionary of URL to HTML content
            html_contents = {
                url: content for url, content in results if content is not None and not isinstance(content, Exception)
            }

            # Process HTML in async executor to avoid blocking event loop
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
