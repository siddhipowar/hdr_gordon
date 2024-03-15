import chronoptics.tof as tof
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
        
        medium = frame_array.flatten()
        plt.hist(medium, bins=50, color='blue', alpha=0.6)
        plt.title('Frame Histogram')
        plt.xlabel('Intensity Value')
        plt.ylabel('Frequency')
        plt.show()

        cv2.imshow('HDR Image Stream', hdr_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    
    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
