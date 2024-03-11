import chronoptics.tof as tof
import numpy as np
import cv2
from typing import List

# Initialize camera
cam = tof.KeaCamera(serial="203001c")

# Select BGR stream only
tof.selectStreams(cam, [tof.FrameType.BGR])

# Start camera
cam.start()

# Create HDR merger from OpenCV
merge_mertens = cv2.createMergeMertens()

# Define different integration times
integration_times = [tof.IntegrationTime.SHORT, tof.IntegrationTime.MEDIUM, tof.IntegrationTime.LONG]

# Loop to stream HDR frames
while cam.isStreaming():
    bgr_frames = []

    # Capture BGR frames at different integration times
    for integration_time in integration_times:
        # Set integration time
        user_config = tof.UserConfig()
        user_config.setIntegrationTime(integration_time)
        camera_config = user_config.toCameraConfig(cam)
        cam.setCameraConfig(camera_config)

        # Get frames
        frame = cam.getFrames()

        # Convert the BGR frame to a numpy array and apply flipping for correct orientation
        bgr_image = np.flipud(np.flip(np.array(frame), axis=2))
        bgr_frames.append(bgr_image)

    # Merge BGR frames into HDR image
    hdr_image = merge_mertens.process(bgr_frames)

    # Convert HDR image to 8-bit for display (simple scaling)
    hdr_display = np.clip(hdr_image * 255, 0, 255).astype('uint8')

    # Show the HDR image using OpenCV
    cv2.imshow('HDR Video Stream', hdr_display)

    # Use cv2.waitKey() for a small delay and to capture a key press; this allows the image to be refreshed.
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the loop
        break

# Release resources when done
cam.stop()
cv2.destroyAllWindows()
