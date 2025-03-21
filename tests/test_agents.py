from agents.research_agent import ResearchAgent

def test_research_agent_run():
    agent = ResearchAgent()
    out = agent.run('test query')
    assert 'query' in out
    assert out['query'] == 'test query'
