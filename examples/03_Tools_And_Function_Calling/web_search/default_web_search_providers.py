from duckduckgo_search import DDGS

from web_search_interfaces import WebSearchProvider


class DDGWebSearchProvider(WebSearchProvider):

    def search_web(self, search_query: str):
        results = DDGS().text(search_query, region='wt-wt', safesearch='off', max_results=4)
        return [res["href"] for res in results]
