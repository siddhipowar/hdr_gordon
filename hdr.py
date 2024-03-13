import chronoptics.tof as tof
import numpy as np
import cv2

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

def process_frames(frames):
    """
    Process the short and long integration time frames to create an HDR-like image
    based on specified algorithm rules, using efficient NumPy operations.
    """
    # Convert frames to 16-bit to prevent overflow during processing
    short_frame = frames[0].astype(np.float32)
    long_frame = frames[1].astype(np.float32)

    # Initialize an HDR image array
    hdr_image = np.zeros_like(short_frame, dtype=np.uint16)

    # Apply conditions
    # Condition a) If one pixel is above 255 or 0 and the other is valid, then use the other's range & intensity data
    valid_short = (short_frame > 0) & (short_frame <= 255)
    valid_long = (long_frame > 0) & (long_frame <= 255)

    hdr_image = np.where(valid_short & ~valid_long, short_frame, hdr_image)
    hdr_image = np.where(valid_long & ~valid_short, long_frame, hdr_image)

    # Conditions b), d), e), and f) with c) overarching intensity scaling
    hdr_image = np.where((long_frame > 200) & valid_short, short_frame, hdr_image)
    hdr_image = np.where((short_frame < 3) & valid_long, long_frame * 20, hdr_image)

    hdr_image = np.where(long_frame == 255, 240, hdr_image)  # 12 on the short_frame scaled up as per e)
    hdr_image = np.where(short_frame == 1, 20, hdr_image)  # f)

    hdr_image_rescaled = (hdr_image / np.max(hdr_image)) * 5100
    hdr_image_rescaled = np.clip(hdr_image_rescaled, 0, 5100).astype(np.uint16)

    return hdr_image_rescaled

def main():
    serial = "203001c"
    cam = tof.KeaCamera(serial=serial)
    # Select INTENSITY stream only
    tof.selectStreams(cam, [tof.FrameType.INTENSITY])
    cam.start()
    
    while cam.isStreaming():

        integration_times = [tof.IntegrationTime.SHORT, tof.IntegrationTime.LONG]
        frames = []

        # Assuming you have a way to set integration times outside this loop since
        # changing settings on-the-fly isn't supported. For demonstration, we capture frames sequentially.
        for integration_time in integration_times:
            frame = capture_frame(cam)
            frames.append(frame)
            # Combine frames into an HDR image
        hdr_image = process_frames(frames)

        hdr_display = np.clip(hdr_image, 0, 255).astype('uint8')
        
        
        cv2.imshow('HDR Image Stream', hdr_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    
    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
