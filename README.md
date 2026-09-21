# YOLO Real-Time Object Detection & Tracking

A Streamlit + WebRTC application for real-time webcam object detection and tracking with YOLO and ByteTrack.

## Features

- Live browser webcam
- YOLO object detection
- ByteTrack persistent tracking IDs
- Multi-class detection
- Current per-class counts
- FPS display
- Unique tracked ID count
- Adjustable confidence and IoU thresholds

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown by Streamlit and click **START**.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload:
   - `app.py`
   - `requirements.txt`
   - `README.md`
3. Go to Streamlit Community Cloud.
4. Connect your GitHub account.
5. Choose the repository and `app.py` as the entrypoint.
6. Deploy.
7. Open the HTTPS app URL and allow camera permission.

The app uses WebRTC because a remote cloud process cannot access the user's physical webcam with `cv2.VideoCapture(0)`.

## Important deployment note

WebRTC normally needs HTTPS for browser camera access and may require STUN/TURN connectivity depending on the network. Streamlit Community Cloud serves apps over HTTPS. If video does not connect from a particular network, a TURN server may be necessary.

## Model

The app loads `yolo26n.pt` through Ultralytics. If your installed Ultralytics version does not provide that model, change it in `app.py` to a compatible pretrained model such as `yolo11n.pt`.

For small objects such as cell phones, lower confidence can help, but may increase false detections.

## Architecture

Webcam
→ WebRTC
→ Streamlit server
→ YOLO detection
→ ByteTrack tracking
→ annotated live video
