from agno.agent import Agent
from agno.models.google import Gemini

agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp", search=True,api_key="AIzaSyAapQ7JFW2GhpR5xAoil7OjAOprHxqDIF4"),
    show_tool_calls=True,
    markdown=True,
)

agent.print_response("What is AI")