from .search_agent import SearchAgent
from .verification_agent import VerificationAgent

class ResearchAgent:
    def __init__(self):
        self.search = SearchAgent()
        self.verify = VerificationAgent()

    def run(self, query: str):
        hits = self.search.run(query)
        verified = self.verify.run(hits)
        return {
            "query": query,
            "hits": hits,
            "verified": verified
        }
