# Real-Time-Height-Object-Measurer

A real-time PyQt5 application to estimate **human height**, **weight**, and **object dimensions** from a webcam feed using **MediaPipe** and **Faster R-CNN**.

---

##  Features

-  Human height & weight estimation using pose landmarks  
-  Object detection with width, height, and volume calculation  
-  Smooth silhouette outline with body segmentation  
-  Uses AI (MediaPipe + Faster R-CNN) for live analysis  
-  Real-time video feed via webcam or DroidCam  
-  Interactive calibration wizard for accuracy

---

##  Calibration System

This code is **calibrated to a fixed frame setup**. It calculates **pixels per cm** by asking the user to click on two points on a known object (e.g., A4 paper = 210mm width).

- Calibration is crucial for accurate real-world measurements.
- The calibration values **vary for different cameras and positions**, so always calibrate once before using.
- Use the **Calibrate** button to input your known object’s height and mark its top and bottom on the frame.
- The measured values like human height, weight, and object volume are based entirely on this pixel/cm ratio.

---

