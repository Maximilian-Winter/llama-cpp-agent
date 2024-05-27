from duckduckgo_search import DDGS
from googlesearch import search

from .web_search_interfaces import WebSearchProvider


class DDGWebSearchProvider(WebSearchProvider):

    def search_web(self, search_query: str, num_results: int):
        results = DDGS().text(search_query, region='wt-wt', safesearch='off', max_results=num_results)
        return [res["href"] for res in results]


class GoogleWebSearchProvider(WebSearchProvider):
    def search_web(self, query: str, num_results: int):
        """Searches the web using Google and returns a list of URLs."""
        try:
            # Only return the top 5 results for simplicity
            return list(search(query, num_results=num_results))
        except Exception as e:
            return f"An error occurred during Google search: {str(e)}"
