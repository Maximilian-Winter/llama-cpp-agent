import abc


class WebCrawler(abc.ABC):
    @abc.abstractmethod
    def get_website_content_from_url(self, url: str):
        """Get the website content from an url."""
        pass


class WebSearchProvider(abc.ABC):
    @abc.abstractmethod
    def search_web(self, query: str, number_of_results: int):
        """Searches the web and returns a list of urls of the result"""
        pass