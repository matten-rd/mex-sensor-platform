# Import the InferencePipeline object
from inference import InferencePipeline
# Import the built in render_boxes sink for visualizing results
from inference.core.interfaces.stream.sinks import render_boxes

import os
from dotenv import load_dotenv

# load api key from .env file using python-dotenv
load_dotenv()
API_KEY = os.environ.get("ROBOFLOW_API_KEY")

# initialize a pipeline object
pipeline = InferencePipeline.init(
    model_id="forest-rock-detection/1", # Roboflow model to use
    video_reference=0, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    on_prediction=render_boxes, # Function to run after each prediction,
    api_key=API_KEY
)
pipeline.start()
pipeline.join()
