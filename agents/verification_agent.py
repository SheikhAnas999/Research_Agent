from .base_agent import BaseAgent

class VerificationAgent(BaseAgent):
    def run(self, data):
        # placeholder verification (fact-check) logic
        return {"verified": True, "data": data}
