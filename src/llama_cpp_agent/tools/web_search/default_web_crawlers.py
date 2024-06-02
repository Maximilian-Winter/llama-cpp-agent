import base64
import json
from datetime import datetime

import requests

from .web_search_interfaces import WebCrawler
from trafilatura import fetch_url, extract
from readability import Document
from bs4 import BeautifulSoup
import httpx


class TrafilaturaWebCrawler(WebCrawler):
    def get_website_content_from_url(self, url: str) -> str:
        """
        Get website content from a URL using Selenium and BeautifulSoup for improved content extraction and filtering.

        Args:
            url (str): URL to get website content from.

        Returns:
            str: Extracted content including title, main text, and tables.
        """

        try:
            downloaded = fetch_url(url)

            result = extract(downloaded, include_formatting=True, include_links=True, output_format='json', url=url)

            if result:
                result = json.loads(result)
                return f'=========== Website Title: {result["title"]} ===========\n\n=========== Website URL: {url} ===========\n\n=========== Website Content ===========\n\n{result["raw_text"]}\n\n=========== Website Content End ===========\n\n'
            else:
                return ""
        except Exception as e:
            return f"An error occurred: {str(e)}"


class BeautifulSoupWebCrawler(WebCrawler):
    def get_website_content_from_url(self, url: str) -> str:
        """
        Get website content from a URL using requests and BeautifulSoup for HTML parsing.

        Args:
            url (str): URL to get website content from.

        Returns:
            str: Extracted content including title and main text.
        """
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            title = soup.find('title').text if soup.find('title') else "No title found"
            body = soup.get_text()

            return f'=========== Website Title: {title} ===========\n\n=========== Website URL: {url} ===========\n\n=========== Website Content ===========\n\n{body}\n\n=========== Website Content End ===========\n\n'
        except Exception as e:
            return f"An error occurred: {str(e)}"


class ReadabilityWebCrawler(WebCrawler):
    def get_website_content_from_url(self, url: str) -> str:
        """
        Get website content from a URL using requests and BeautifulSoup for HTML parsing.

        Args:
            url (str): URL to get website content from.

        Returns:
            str: Extracted content including title and main text.
        """
        try:

            response = requests.get(url)
            doc = Document(response.content)

            title = doc.title()
            body = doc.summary()

            return f'=========== Website Title: {title} ===========\n\n=========== Website URL: {url} ===========\n\n=========== Website Content ===========\n\n{body}\n\n=========== Website Content End ===========\n\n'
        except Exception as e:
            return f"An error occurred: {str(e)}"


