import cv2
import threading
from pathlib import Path
from agno.agent import Agent
from agno.media import Video
from agno.models.google import Gemini

# Initialize Gemini AI agent
agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp",api_key="AIzaSyAapQ7JFW2GhpR5xAoil7OjAOprHxqDIF4"),
    markdown=True,
    instructions=[
         "You are an AI agent that detects construction site safety compliance.",
        "Identify and describe any safety objects such as safety helmets, safety jackets, gloves, boots, and other protective gear.",
        "For each person in the video, indicate whether they are wearing proper safety equipment.",
        "Return the response in the format: 'Helmet: Yes/No, Safety Jacket: Yes/No, Gloves: Yes/No, Boots: Yes/No'."
    ],
)

# Video file path
video_path = str(Path(__file__).parent.joinpath("safety.mp4"))

# Open video with OpenCV
cap = cv2.VideoCapture(video_path)

# Ensure video is opened
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()


def analyze_video():
    """Send video to Gemini AI for analysis while playing."""
    print("Sending video to Gemini AI for analysis...")
    
    # Send video to Gemini AI
    response = agent.print_response(
         "Analyze this video and detect the presence of safety helmets, safety jackets, gloves, boots, and other protective gear.",
        videos=[Video(filepath=video_path)],
    )
    
    print("\n🔍 Gemini AI Response:")
    print(response)  # Print AI's analysis


# Run AI analysis in a separate thread
thread = threading.Thread(target=analyze_video)
thread.start()

# Play the video using OpenCV
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame=cv2.resize(frame,(1020,600))
    if not ret:
        break  # Stop if video ends

    cv2.imshow("Playing Video", frame)  # Display the video

    if cv2.waitKey(60) & 0xFF == ord("q"):  # Press 'q' to exit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Wait for AI response thread to finish
thread.join()
