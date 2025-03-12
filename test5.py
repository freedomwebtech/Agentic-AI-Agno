import cv2
import threading
import time
from pathlib import Path
from agno.agent import Agent
from agno.media import Image
from agno.models.google import Gemini

# Initialize Gemini AI agent
agent = Agent(
    model=Gemini(id="gemini-2.0-flash", api_key="AIzaSyAapQ7JFW2GhpR5xAoil7OjAOprHxqDIF4"),
    markdown=True,
    instructions=[
        "You are an AI agent that detects construction site safety compliance.",
        "Identify and describe any safety objects such as safety helmets, safety jackets, gloves, boots, and other protective gear.",
        "For each person in the video, indicate whether they are wearing proper safety equipment.",
        "Return the response in the format: 'Helmet: Yes/No, Safety Jacket: Yes/No, Gloves: Yes/No, Boots: Yes/No'."
    ],
)

# Set up webcam
cap = cv2.VideoCapture(0)  # 0 for the default webcam
image_path = str(Path(__file__).parent.joinpath("latest_frame.jpg"))

# Global variable for latest frame
latest_frame = None
lock = threading.Lock()  # Lock to synchronize frame access

def analyze_frame():
    """Continuously sends the latest frame to Gemini AI for processing."""
    global latest_frame

    while True:
        time.sleep(1)  # Adjust interval to balance performance
        
        with lock:
            if latest_frame is None:
                continue
            frame_copy = latest_frame.copy()

        # Save frame as an image
        cv2.imwrite(image_path, frame_copy)

        # Send image to Gemini AI for analysis
        response = agent.print_response(
             "Analyze this video and detect the presence of safety helmets, safety jackets, gloves, boots, and other protective gear.",
            images=[Image(filepath=image_path)],
        )

        print("\n🔍 Gemini AI Response:")
        print(response)


# Start AI processing in a separate thread
ai_thread = threading.Thread(target=analyze_frame, daemon=True)
ai_thread.start()

# Display live webcam feed
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))

    with lock:
        latest_frame = frame.copy()  # Store latest frame for AI processing

    cv2.imshow("🎥 Live Webcam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
