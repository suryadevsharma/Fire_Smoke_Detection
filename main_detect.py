from ultralytics import YOLO
import cv2

# 1. Load model
# Make sure the path is correct
model = YOLO("models/best.pt")

def run_on_image(img_path):
    # Read image
    img = cv2.imread(img_path)

    if img is None:
        print("Error: Could not read image. Check the path.")
        return

    # Run inference
    results = model(img)

    # Draw bounding boxes
    annotated = results[0].plot()

    # Show the output
    cv2.imshow("Fire & Smoke Detection - Image", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_on_video(video_path=0):
    # 0 = webcam, or give video file path
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or cannot read frame.")
            break

        # Run inference
        results = model(frame)

        # Draw boxes
        annotated = results[0].plot()

        # Display frame
        cv2.imshow("Fire & Smoke Detection - Video", annotated)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 👉 To test on an image, put an image in data/ and uncomment this:
     run_on_image("data/test_fire.jpg")

    # 👉 To test on webcam (default):
    #run_on_video(0)
