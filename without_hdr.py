import chronoptics.tof as tof
import numpy as np
import cv2

def main():
    serial = "202004d"
    cam = tof.KeaCamera(serial=serial)
    # Select INTENSITY stream only
    tof.selectStreams(cam, [tof.FrameType.INTENSITY])
    user_config = tof.UserConfig()
    user_config.setIntegrationTime(tof.IntegrationTime.MEDIUM)
    camera_config = user_config.toCameraConfig(cam)
    cam.setCameraConfig(camera_config)
    cam.start()
    print(cam.getCalibration().getCalibratedFrequencies())
    
    while cam.isStreaming():
        frame = cam.getFrames()
        frame_array = np.asarray(frame[0], dtype=np.uint8)

        hdr_display = np.clip(frame_array, 0, 255).astype('uint8')
        
        cv2.imshow('HDR Image Stream', hdr_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    
    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
