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
    except Exception:
        return False


def parse_html(html_content: Optional[str]) -> str:
    """Parse HTML content and extract text with hyperlinks in markdown format.

    Extracts meaningful content while filtering out:
    - Navigation elements
    - Advertisements
    - Social media widgets
    - References and citations
    - Footer content
    - Redundant links
    - Boilerplate text

    Hyperlinks are formatted as [link text](URL).
    """
    if not html_content:
        return ""

    try:
        document = html5lib.parse(html_content)
        result = []
        seen_texts = set()  # To avoid duplicate lines

        # Elements that typically contain non-essential content
        SKIP_CLASSES = {
            'nav', 'navigation', 'menu', 'footer', 'header', 'sidebar', 
            'widget', 'ad', 'advertisement', 'social', 'share', 'related',
            'comments', 'reference', 'footnote', 'citation', 'copyright'
        }
        SKIP_IDS = {
            'nav', 'navigation', 'menu', 'footer', 'header', 'sidebar',
            'references', 'footnotes', 'citations', 'related-content'
        }

        def should_skip_element(elem) -> bool:
            """Determine if the element should be skipped."""
            # Skip non-content tags
            if elem.tag in [
                "{http://www.w3.org/1999/xhtml}script",
                "{http://www.w3.org/1999/xhtml}style",
                "{http://www.w3.org/1999/xhtml}noscript",
                "{http://www.w3.org/1999/xhtml}iframe",
                "{http://www.w3.org/1999/xhtml}svg",
            ]:
                return True

            # Skip based on class attribute
            class_attr = elem.get("class", "")
            if class_attr:
                classes = set(class_attr.lower().split())
                if classes & SKIP_CLASSES:
                    return True

            # Skip based on id attribute
            id_attr = elem.get("id", "")
            if id_attr and id_attr.lower() in SKIP_IDS:
                return True

            # Skip elements with only whitespace text
            if not any(text.strip() for text in elem.itertext()):
                return True

            return False

        def is_meaningful_link(href: str, text: str) -> bool:
            """Check if a link is meaningful and worth keeping."""
            if not href or not text:
                return False

            # Filter out common non-content URL patterns
            if any(pattern in href.lower() for pattern in [
                'javascript:', '#', 'mailto:', 'tel:',
                '/tag/', '/category/', '/author/', '/page/',
                'twitter.com', 'facebook.com', 'linkedin.com',
                'instagram.com', '/feed/', '/rss/',
                'policy', 'terms', 'privacy', 'cookie'
            ]):
                return False

            # Skip very short link text
            if len(text.strip()) < 3:
                return False

            return True

        def add_text(text: str, depth: int):
            """Normalize text and add to results if meaningful."""
            norm_text = ' '.join(text.strip().split())
            if norm_text and norm_text not in seen_texts and len(norm_text) > 2:
                result.append("  " * depth + norm_text)
                seen_texts.add(norm_text)

        def process_element(elem, depth=0):
            """Recursively process an element and its children."""
            if should_skip_element(elem):
                return

            # Process anchor tags differently for markdown formatting
            if elem.tag == "{http://www.w3.org/1999/xhtml}a":
                href = elem.get("href")
                link_text = elem.text or ""
                norm_text = ' '.join(link_text.strip().split())
                if href and is_meaningful_link(href, norm_text):
                    markdown_link = f"[{norm_text}]({href})"
                    if markdown_link not in seen_texts:
                        result.append("  " * depth + markdown_link)
                        seen_texts.add(markdown_link)
            else:
                if elem.text:
                    add_text(elem.text, depth)

            # Recursively process child elements
            for child in elem:
                process_element(child, depth + 1)

            # Process tail text
            if elem.tail:
                add_text(elem.tail, depth)

        # Try to target the main content area
        main_content = document.find(".//{http://www.w3.org/1999/xhtml}main")
        article = document.find(".//{http://www.w3.org/1999/xhtml}article")
        content = document.find(".//*[@id='content']")

        if main_content is not None:
            process_element(main_content)
        elif article is not None:
            process_element(article)
        elif content is not None:
            process_element(content)
        else:
            # Fallback to the body or the whole document
            body = document.find(".//{http://www.w3.org/1999/xhtml}body")
            if body is not None:
                process_element(body)
            else:
                process_element(document)

        # Filter out common unwanted noise patterns
        filtered_result = []
        noise_patterns = [
            'var ', 'function()', '.js', '.css', 'google-analytics', 'disqus',
            '{', '}', 'cookie', 'subscribe', 'newsletter', 'sign up',
            'download', 'click here', 'read more', 'learn more'
        ]
        for line in result:
            lower_line = line.lower()
            if any(pattern in lower_line for pattern in noise_patterns):
                continue
            clean_line = line.strip()
            if len(clean_line) < 3 or clean_line.replace(' ', '').isdigit():
                continue
            filtered_result.append(clean_line)

        return '\n'.join(filtered_result)

    except Exception as e:
        logger.error(f"Error parsing HTML: {str(e)}")
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
