# JJK Cursed Vision
### Real-Time Anime Hand Gesture Recognition System

![JJK Demo](https://giffiles.alphacoders.com/220/220766.gif)

> "Throughout Heaven and Earth, I alone am the honored one."

A real-time AI-powered hand gesture recognition system inspired by Jujutsu Kaisen, where your hand signs trigger cinematic cursed techniques with immersive 3D particle effects in the browser.

---

## Features

- Real-time hand tracking using webcam  
- Anime-style cursed technique visual effects  
- 3D particle rendering using Three.js  
- Live gesture-based interaction  
- MediaPipe-based hand landmark detection  

---

## How It Works

The system combines Python-based gesture detection with a browser-based visual engine:

- Webcam feed is captured using OpenCV  
- MediaPipe detects 21 hand landmarks in real time  
- Gesture logic analyzes finger positions and patterns  
- Each gesture is mapped to a specific cursed technique  
- Frontend (Three.js) renders animated particle effects based on detected gestures  

---

## Supported Gestures

| Gesture | Technique | Description |
|--------|----------|------------|
| 🤏🏻 Pinch | Hollow Purple | High-energy destructive blast |
| ✌🏻 Two fingers up | Red | Reverse cursed technique |
| 🤞🏻 Cross fingers | Infinite Void | Domain expansion |
| 🖐🏻 (thumb down) | Malevolent Shrine | Sukuna’s domain |

---

## Tech Stack

Backend:
- Python 3.9+
- OpenCV  
- MediaPipe  
- NumPy  

Frontend:
- HTML, CSS, JavaScript  
- Three.js  
- WebGL  

---

## Project Structure

.
├── main.py  
├── hand_tracker.py  
├── gesture_utils.py  
├── index.html  
├── requirements.txt  

---

## Installation

Clone the repository:

git clone https://github.com/xianceor/jjk-cursed-vision.git  
cd jjk-cursed-vision  

Create virtual environment:

Windows:
python -m venv venv  
venv\Scripts\activate  

macOS/Linux:
python3 -m venv venv  
source venv/bin/activate  

Install dependencies:

pip install -r requirements.txt  

---

## How to Run

Run the backend:

python main.py  

Then run the frontend:

Recommended method (Live Server):

- Open the project folder in VS Code  
- Install "Live Server" extension  
- Right-click on index.html  
- Click "Open with Live Server"  

Alternative method:

- Open index.html directly in browser (some features may not work properly)  

---

## System Requirements

Minimum:
- 4GB RAM  
- Webcam  

Recommended:
- 8GB+ RAM  
- GPU (for smooth rendering)  
- Chrome or Edge browser  

---

## Troubleshooting

- Webcam not working → Close other apps using camera  
- Low FPS → Reduce resolution in Python  
- Gesture not detected → Improve lighting and hand visibility  
- Effects not showing → Use Chrome with Live Server  
- Black screen → Check WebGL support  

---

## Future Improvements

- Machine learning based gesture classification  
- Sound effects for techniques  
- Multi-hand combo system  
- Mobile browser support  
- Interactive gameplay elements  

---

## License

MIT License  

---

## Author

Built with caffeine, code, and cursed energy  

---

Star this repo if you felt like Gojo for even one second
