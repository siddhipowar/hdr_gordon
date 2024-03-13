import chronoptics.tof as tof
import numpy as np
import cv2
import time

def set_camera_integration_time(cam, integration_time):
    # Configure the camera with the new integration time without restarting the stream
    user_config = tof.UserConfig()
    user_config.setIntegrationTime(integration_time)
    camera_config = user_config.toCameraConfig(cam)
    cam.setCameraConfig(camera_config)

def capture_frame(cam):
    # Capture a single frame without stopping the camera
    frames = cam.getFrames()
    frame_array = np.asarray(frames[0], dtype=np.uint8)
    return frame_array

import numpy as np

import numpy as np
import cv2  # Assuming OpenCV is available for alignment, unsharp masking, and CLAHE

def process_frames(frames):
    # Assuming frames[0] and frames[1] are already aligned; if not, align them first.
    short_frame = frames[0].astype(np.float32)
    long_frame = frames[1].astype(np.float32)

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
    serial = "203001c"
    cam = tof.KeaCamera(serial=serial)
    # Select INTENSITY stream only
    tof.selectStreams(cam, [tof.FrameType.INTENSITY])
    cam.start()
    
    while cam.isStreaming():

        time.sleep(2)
        integration_times = [tof.IntegrationTime.SHORT, tof.IntegrationTime.LONG]
        frames = []

        # Assuming you have a way to set integration times outside this loop since
        # changing settings on-the-fly isn't supported. For demonstration, we capture frames sequentially.
        for integration_time in integration_times:
            user_config = tof.UserConfig()
            time.sleep(2)
            user_config.setIntegrationTime(integration_time)
            camera_config = user_config.toCameraConfig(cam)
            cam.setCameraConfig(camera_config)
            time.sleep(2)
            frame = capture_frame(cam)
            frames.append(frame)
            # Combine frames into an HDR image
        hdr_image = process_frames(frames)
        
        cv2.imshow('HDR Image Stream', hdr_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
