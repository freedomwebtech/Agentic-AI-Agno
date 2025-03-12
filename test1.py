from pathlib import Path
from agno.agent import Agent
from agno.media import Image
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.google import Gemini  # ✅ Use Gemini, not OpenAIChat

# Initialize the agent with Gemini Vision model
agent = Agent(
    model=Gemini(id="gemini-2.0-flash", api_key=""),  # ✅ Correct placement of API key
    agent_id="image-to-text",
    name="Image to Text Agent",
    tools=[DuckDuckGoTools()],
    markdown=True,
    debug_mode=False,
    show_tool_calls=True,
    instructions=[
        "You are an AI agent that can generate text descriptions based on an image.",
        "You have to return a text response describing the image.",
    ],
)

# Load the image
image_path = Path(__file__).parent.joinpath("s.jpg")

# Ask Gemini to describe the image
agent.print_response(
    "what is in image give me details give me  compnay name",
    images=[Image(filepath=str(image_path))],  # ✅ Ensure image path is passed as a string
    stream=True,
)
