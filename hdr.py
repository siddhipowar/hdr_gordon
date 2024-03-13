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
    # Assuming frames[0] is the short exposure and frames[1] is the long exposure
    # Convert frames to 32-bit float for processing
    short_frame = frames[0].astype(np.float32)
    long_frame = frames[1].astype(np.float32)
    
    # Apply initial conditions to handle saturated and underexposed pixels
    valid_short = (short_frame > 0) & (short_frame <= 255)
    valid_long = (long_frame > 0)
    hdr_image = np.where(valid_short & ~valid_long, short_frame, long_frame)
    hdr_image = np.where(valid_long & (long_frame <= 255), long_frame, hdr_image)
    
    # Apply special conditions based on the intensity
    hdr_image = np.where((long_frame > 200) & valid_short, short_frame, hdr_image)
    hdr_image = np.where((short_frame < 3) & valid_long, long_frame * 20, hdr_image)

    # Weighted average combination of short and long frames for the rest of the pixels
    # Maximum intensity for long exposure images, adjust if different
    max_intensity_long = 5100
    # Define weight threshold for HDR combination
    weight_threshold = 200
    # Normalize frames to the same intensity scale
    short_norm = short_frame / 255
    long_norm = hdr_image / max_intensity_long
    # Calculate the weights for the long exposure based on intensity
    weights_long = np.clip(long_norm * (1 / weight_threshold), 0, 1)
    weights_short = 1 - weights_long
    # Blend the two exposures based on weights
    hdr_blended = short_norm * weights_short + long_norm * weights_long

    # Normalize the blended HDR image to utilize the full dynamic range
    hdr_normalized = (hdr_blended - np.min(hdr_blended)) / (np.max(hdr_blended) - np.min(hdr_blended))
    
    return hdr_normalized

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

        hdr_display = (hdr_image *255).astype('uint8')
        
        
        cv2.imshow('HDR Image Stream', hdr_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    
    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
