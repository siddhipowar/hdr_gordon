import chronoptics.tof as tof
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

def plot_histograms(short_frame, long_frame, hdr_image):
    short = short_frame.flatten()
    long = long_frame.flatten()
    hdr = hdr_image.flatten()

    plt.figure(figsize=(18, 6))

    # Short Frame Histogram
    plt.subplot(1, 3, 1)
    plt.hist(short, bins=50, color='blue', alpha=0.6)
    plt.title('Short Frame Histogram')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')

    # Long Frame Histogram
    plt.subplot(1, 3, 2)
    plt.hist(long, bins=50, color='red', alpha=0.6)
    plt.title('Long Frame Histogram')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')

    # HDR Image Histogram
    plt.subplot(1, 3, 3)
    plt.hist(hdr, bins=50, color='green', alpha=0.6)
    plt.title('HDR Image Histogram')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def capture_frame(cam):
    # Capture a single frame without stopping the camera
    frames = cam.getFrames()
    frame_array = np.asarray(frames[0], dtype=np.uint8)
    return frame_array


def process_frames(frames):

    short_frame = frames[0].astype(np.float64)
    long_frame = frames[1].astype(np.float64)

    short_frame /= 255.0
    long_frame /= 5100.0

    # Update mask thresholds based on image analysis (optional step not shown).

    mask_saturated_short = short_frame > 0.9
    mask_shadow_long = long_frame < 0.1

    combined = np.where(mask_saturated_short, long_frame, short_frame)
    combined = np.where(mask_shadow_long, short_frame, combined)

    combined_normalized = (combined - combined.min()) / (combined.max() - combined.min())

    sigmoid_strength = 6
    weights = 1 / (1 + np.exp(sigmoid_strength * (combined_normalized - 0.5)))
    weights = np.clip(weights, 0.3, 0.7)

    hdr_blended = short_frame * (1 - weights) + long_frame * weights

    hdr_normalized = (hdr_blended - hdr_blended.min()) / (hdr_blended.max() - hdr_blended.min())

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    hdr_enhanced = clahe.apply(np.uint8(hdr_normalized*255)) / 255.0  # Assuming the image is grayscale; adjust if not.

    hdr_display = np.clip(hdr_enhanced * 255, 0, 255).astype(np.uint8)

    return hdr_display


def main():
    serial = "202004d"
    cam = tof.KeaCamera(serial=serial)
    # Select INTENSITY stream only
    tof.selectStreams(cam, [tof.FrameType.INTENSITY])
    cam.start()
    
    while cam.isStreaming():

        time.sleep(2)
        # integration_times = [tof.IntegrationTime.SHORT, tof.IntegrationTime.LONG]
        frames = []

        config = cam.getCameraConfig()
        config.reset()
        config.setIntegrationTime(0, [50, 50, 50, 50])
        cam.setCameraConfig(config)
        time.sleep(2)
        frame1 = np.copy(capture_frame(cam))

        config = cam.getCameraConfig()
        config.reset()
        config.setIntegrationTime(0, [1000, 1000, 1000, 1000])
        cam.setCameraConfig(config)
        time.sleep(2)
        frame2 = capture_frame(cam)

        frames = [frame1, frame2]

        # Combine frames into an HDR image
        hdr_image = process_frames(frames)

        plot_histograms(frame1, frame2, hdr_image)
        
        cv2.imshow('HDR Image Stream', hdr_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
