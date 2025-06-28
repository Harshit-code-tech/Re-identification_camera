Player Tracking and Re-Identification
Overview
This project implements a player tracking and re-identification pipeline for a 15-second video clip. It uses a fine-tuned YOLOv11 model for player detection, the Norfair library for tracking, and histogram-based appearance matching for re-identification to maintain consistent player identities across frames, even when players exit and re-enter the scene.
Objectives

Detect players in a video using YOLOv11.
Track players using Norfair with Euclidean distance-based tracking.
Re-identify players who re-enter the frame using HSV histogram comparison.
Generate two output videos: one with tracking IDs and one with re-identified IDs.
Log ID switches and plot histogram similarity scores for analysis.

Prerequisites

Environment: Kaggle notebook or a Python environment with GPU support (recommended for YOLOv11).
Dependencies:
Python 3.8+
Libraries: ultralytics, opencv-python-headless, norfair, matplotlib, pandas


Input Files:
Video: 15_sec_video.mp4 (located at /kaggle/input/15-sec-video/15_sec_video.mp4)
YOLOv11 model: best.pt (located at /kaggle/input/best/pytorch/default/1/best.pt)



Installation

Clone or download this repository.
Install dependencies:pip install ultralytics opencv-python-headless norfair matplotlib pandas --quiet


Ensure the input video and model files are in the correct paths (modify paths if running locally).

Usage

Run the Jupyter Notebook:
Open player_tracking_reid.ipynb in a Jupyter environment (e.g., Kaggle, JupyterLab).
Execute all cells sequentially to process the video, generate outputs, and create visualizations.


Outputs:
Videos:
tracked_output_with_head.mp4: Video with tracking IDs and head regions.
reid_output.mp4: Video with re-identified IDs and head regions.


Logs and Plots:
id_switch_log.csv: Logs ID switches with frame number, tracker ID, old ID, new ID, and similarity score.
reid_scores_plot.png: Plot of histogram similarity scores over time.
Sample frames: Saved every 30 frames as sample_frame_{frame_num}.jpg.




Configuration:
Adjust parameters in the notebook’s "Configuration" section (e.g., distance_threshold, num_reference_frames, match_threshold) if needed.



Methodology

Detection: Uses a fine-tuned YOLOv11 model to detect players with a confidence threshold of 0.3. Only the most confident detection is kept per frame, assuming a single player.
Tracking: Norfair tracks players using Euclidean distance between bounding box centers, with a distance threshold of 150 pixels.
Re-Identification: Builds a reference gallery from the first 50 frames. For later frames, compares HSV histograms of detected players against the gallery, assigning the best-matching ID if the similarity score exceeds 0.85.
Visualization: Draws bounding boxes, tracking/re-identified IDs, and head regions (top 25% of the bounding box) on output videos. Logs ID switches and plots similarity scores.

File Structure
├── player_tracking_reid.ipynb  # Main Jupyter notebook
├── /kaggle/input/15-sec-video/15_sec_video.mp4  # Input video
├── /kaggle/input/best/pytorch/default/1/best.pt  # YOLOv11 model
├── /kaggle/working/  # Output directory
│   ├── reid_output.mp4  # Re-identification video
│   ├── tracked_output_with_head.mp4  # Tracking video
│   ├── id_switch_log.csv  # ID switch log
│   ├── reid_scores_plot.png  # Similarity score plot
│   ├── sample_frame_*.jpg  # Sample frames

Notes

The code assumes a single player in the video, as per the task. For multiple players, remove the single-detection filter or enhance the re-identification logic.
The histogram-based re-identification may be sensitive to lighting changes. Consider deep learning-based re-identification for robustness.
Ensure the /kaggle/working/ directory is writable and has sufficient space for outputs.
Paths are set for a Kaggle environment. Modify them for local execution (e.g., ./input/15_sec_video.mp4).

Troubleshooting

Video not opening: Verify the video path and format (must be MP4).
Model errors: Ensure the YOLOv11 model file exists and is compatible with the ultralytics library.
No detections: Check if the player class is in model.names. Adjust player_class_id if necessary.
Output issues: Confirm the output directory is writable and the video writer codec (mp4v) is supported.

License
This project is provided for educational purposes. Ensure you have permission to use the input video and model files.
