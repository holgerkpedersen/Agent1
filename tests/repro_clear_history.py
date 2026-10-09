import pytest
import asyncio
from unittest.mock import MagicMock, patch
from agent import Agent

@pytest.mark.asyncio
async def test_clear_history_clears_conversation():
    # We need to mock LLMClient because Agent.__init__ might try to use it 
    # or we need to provide a way for it not to fail during instantiation.
    
    with patch('agent.LLMClient') as MockClient:
        # To avoid the KeyError in get_profile, we'll mock resolve_model and build_transport too
        with patch('agent.resolve_model', return_value="mock-model"), \
             patch('agent.to_windows_path', side_effect=lambda x: x), \
             patch('agent.FullRunGate'), \
             patch('agent.FileSystem'), \
             patch('agent.FileSearcher'), \
             patch('agent.ToolDispatcher'), \
             patch('agent.load_agent_settings', return_value={}):
            
            # Instantiate agent
            agent = Agent(workspace=".")
            
            # Ensure the attribute exists for the test to be meaningful.
            if not hasattr(agent, '_chat_history'):
                agent._chat_history = []

            # Simulate some conversation by manually appending to _chat_history
            agent._chat_history.append({"role": "user", "content": "old message"})
            agent._turn_start_index = 10
            
            # Perform clear_history
            agent.clear_history()
            
            # Verify _chat_history is empty
            assert len(agent._chat_history) == 0, f"Expected _chat_history to be empty, but got {len(agent._chat_history)}"
            
            # Verify _turn_start_index is reset
            assert agent._turn_start_index == 0, f"Expected _turn_start_index to be 0, but got {agent._turn_start_index}"

if __name__ == "__main__":
    pytest.main([__file__])
