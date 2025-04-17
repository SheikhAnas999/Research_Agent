"""Entry point for the ai_research_agent project."""
from agents.research_agent import ResearchAgent
from utils.logger import get_logger

log = get_logger(__name__)

def main():
    agent = ResearchAgent()
    log.info("Starting Research Agent")
    # simple demonstration run
    results = agent.run("Example research query")
    print("Results:", results)

if __name__ == "__main__":
    main()
