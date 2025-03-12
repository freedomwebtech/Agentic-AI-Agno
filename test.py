from agno.agent import Agent
from agno.models.google import Gemini

agent = Agent(
    model=Gemini(id="gemini-2.0-flash", search=True,api_key=""),
    show_tool_calls=True,
    markdown=True,
)

agent.print_response("What is AI")
