# hdr_gordon

HDR Program for Gordon which captures frames of different integration times and combines them to form an HDR image frame for the camera stream

Algorithm:
Initialize camera with serial number and frame type as intensity frames


While camera is streaming 50 us integration time frame and 1000 us integration time frame is captured and passed to process_frames()


process_frames(frame1, frame2): This is the main function that implements HDR. 
- Declare a overexposure threshold value.
- Data in both the frames is converted to a range of 0 - 1 for processing
- ‘mask_overexposed’ is a boolean mask to detect where both short and long frame are overexposed based on the threshold
  mask_overexposed = (short_frame > overexposure_threshold) & (long_frame > overexposure_threshold)
- Gaussian blur is used to reduce noise in the long frame
- Both frames are combined. It uses long frame data when there is overexposure and uses maximum between both frames in other case
- combined[mask_overexposed] = 1.0 - (1.0 - combined[mask_overexposed])**0.5
  Above expression selects all pixels in combined frame that are overexposed. Subtracting that from 1 inverts the pixel values, square root applied after that helps                                 in reducing the intensity of overexposed regions while retaining the detail. Subtracting from 1 again inverts the values again which maps them back to higher end of intensity range but with reduced contrast and preserved details in the overexposed areas.
- Functions applied for enhancing overall contrast, and balancing the lighting across an image


process_frames() returns an HDR image in a displayable format that is fed to the camera stream
