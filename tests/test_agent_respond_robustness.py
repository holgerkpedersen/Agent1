import pytest
from unittest.mock import MagicMock
from agent_core.agent import LLMAgent

class MockLLMClient:
    def chat(self, messages: list[dict[str, str]]) -> str:
        return "mock response"

def test_respond_clears_temp_messages_even_on_failure():
    # Setup agent with a failing client
    client = MagicMock()
    client.chat.side_effect = Exception("LLM Error")
    agent = LLMAgent(llm_client=client)
    
    # Manually inject some temp messages to simulate file context detection
    agent._pending_temp_systems = [{"role": "system", "content": "some context"}]
    
    with pytest.raises(Exception, match="LLM Error"):
        agent.respond("hello")
    
    # Verify that even on failure, temp messages are cleared
    assert len(agent._pending_temp_systems) == 0

def test_respond_appends_to_conversation_on_success():
    client = MockLLMClient()
    agent = LLMAgent(llm_client=client)
    
    response = agent.respond("hello")
    
    assert response == "mock response"
    assert len(agent._conversation) == 2  # user + assistant
    assert agent._conversation[0] == {"role": "user", "content": "hello"}
    assert agent._conversation[1] == {"role": "assistant", "content": "mock response"}

def test_respond_fallback_when_no_client():
    agent = LLMAgent(llm_client=None)
    response = agent.respond("hello")
    
    assert response == "I received your message but have no LLM client available."
    assert len(agent._conversation) == 2
    assert agent._conversation[1] == {"role": "assistant", "content": response}

if __name__ == "__main__":
    pytest.main([__file__])
