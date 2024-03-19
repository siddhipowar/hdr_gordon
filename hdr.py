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

    # Parameters for processing
    overexposure_threshold = 0.9

    short_frame = frames[0].astype(np.float32)
    long_frame = frames[1].astype(np.float32)

    # Normalize the frames
    short_frame /= 255.0
    long_frame /= 255.0

    # Identify overexposed regions in both frames
    mask_overexposed = (short_frame > overexposure_threshold) & (long_frame > overexposure_threshold)

    # Create a smoothed version of the long exposure frame using Gaussian blur
    long_frame_smoothed = cv2.GaussianBlur(long_frame, (5, 5), 0)

    # Blend the original and the smoothed frames
    combined = np.where(mask_overexposed, long_frame_smoothed, np.maximum(short_frame, long_frame))

    # Apply dynamic range compression to the overexposed regions
    combined[mask_overexposed] = 1.0 - (1.0 - combined[mask_overexposed])**0.5

    # Normalize the combined image
    combined_normalized = cv2.normalize(combined, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Apply CLAHE to the normalized combined image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    hdr_enhanced = clahe.apply(np.uint8(combined_normalized * 255.0)) / 255.0  

    # Gamma correction can still be applied if needed
    gamma = 0.8
    combined_gamma_corrected = np.power(hdr_enhanced, gamma)

    # Convert to 8-bit format for display
    hdr_display = np.clip(combined_gamma_corrected * 255, 0, 255).astype(np.uint8)

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
