from duckduckgo_search import DDGS
from googlesearch import search

from .web_search_interfaces import WebSearchProvider


class DDGWebSearchProvider(WebSearchProvider):

    def search_web(self, search_query: str):
        results = DDGS().text(search_query, region='wt-wt', safesearch='off', max_results=4)
        return [res["href"] for res in results]


class GoogleWebSearchProvider(WebSearchProvider):
    def search_web(self, query: str):
        """Searches the web using Google and returns a list of URLs."""
        try:
            # Only return the top 5 results for simplicity
            return list(search(query, num_results=5))
        except Exception as e:
            return f"An error occurred during Google search: {str(e)}"
