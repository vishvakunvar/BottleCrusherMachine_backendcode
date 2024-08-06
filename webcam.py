import cv2
import time

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    frame_counter = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        frame_counter += 1

        # Display the frame
        cv2.imshow('Video Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate frame rate and print it
    end_time = time.time()
    elapsed_time = end_time - start_time
    frame_rate = frame_counter / elapsed_time
    print(f"Average Frame Rate: {frame_rate:.2f} FPS")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
