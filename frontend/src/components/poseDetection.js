// PoseDetection.js
import React, { useRef, useState, useEffect } from 'react';
import axios from 'axios';
import './PoseDetection.css'; // Make sure to create this CSS file

// Define the backend URL as a constant
const BACKEND_URL = 'http://127.0.0.1:5001';

const PoseDetection = () => {
  const videoRef = useRef(null);
  const uploadVideoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [detectionResults, setDetectionResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState('webcam'); // 'webcam' or 'upload'
  const [uploadedVideo, setUploadedVideo] = useState(null);
  const [processingVideo, setProcessingVideo] = useState(false);
  const [processProgress, setProcessProgress] = useState(0);
  const [videoFileDetails, setVideoFileDetails] = useState(null);
  const [error, setError] = useState(null);
  const [realTimeDetection, setRealTimeDetection] = useState(false); // Start disabled
  const [realTimeKeypoints, setRealTimeKeypoints] = useState(null); // Store keypoints
  const [realTimeFrames, setRealTimeFrames] = useState([]); // Store all frames for CSV
  
  // Start webcam when component mounts if in webcam mode
  useEffect(() => {
    if (mode === 'webcam') {
      startWebcam();
    }
    
    // Cleanup function to stop webcam when component unmounts
    return () => {
      const video = videoRef.current;
      if (video && video.srcObject) {
        const tracks = video.srcObject.getTracks();
        tracks.forEach(track => track.stop());
      }
    };
  }, [mode]);
  
  // Add a canvas drawing loop with real-time detection for webcam mode
  useEffect(() => {
    let animationId;
    let lastDetectionTime = 0;
    const detectionInterval = 100; // Detect poses every 100ms
    
    const drawAndDetect = async (timestamp) => {
      if (videoRef.current && videoRef.current.readyState === 4 && canvasRef.current && mode === 'webcam') {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        
        // Set canvas dimensions to match video
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }
        
        // Always draw video frame to canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // If we have keypoints from real-time detection, draw them
        if (realTimeKeypoints) {
          drawKeypoints(realTimeKeypoints, false); // false means don't redraw the video frame
        }
        
        // If real-time detection is enabled and enough time has passed
        if (realTimeDetection && timestamp - lastDetectionTime > detectionInterval) {
          lastDetectionTime = timestamp;
          
          // Capture the current frame
          const imageData = canvas.toDataURL('image/jpeg', 0.8);
          
          // Send to backend without blocking the animation loop
          sendImageToBackendRealTime(imageData);
        }
      }
      
      // Continue the loop
      animationId = requestAnimationFrame(drawAndDetect);
    };
    
    if (mode === 'webcam' && videoRef.current) {
      animationId = requestAnimationFrame(drawAndDetect);
    }
    
    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [mode, realTimeDetection, realTimeKeypoints]); // Add realTimeKeypoints as dependency
  
  const startWebcam = async () => {
    try {
      setError(null);
      const constraints = { 
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user"
        } 
      };
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        
        // Wait for video to be ready
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
        };
      }
      
    } catch (err) {
      console.error("Error accessing webcam:", err);
      setError("Could not access webcam. Please ensure you've granted camera permissions.");
    }
  };
  
  // Send image to backend without blocking UI or showing loading state
  const sendImageToBackendRealTime = async (imageData) => {
    try {
      // Send the image without waiting for response
      const response = await axios.post(`${BACKEND_URL}/detect_pose`, {
        image: imageData,
        save_csv: false // Don't save individual frames to CSV in real-time mode
      }, {
        headers: {
          'Content-Type': 'application/json'
        },
        timeout: 3000 // Shorter timeout for real-time
      });
      
      // If we get results, update the real-time keypoints
      if (response.data && response.data.keypoints) {
        setRealTimeKeypoints(response.data.keypoints);
        
        // Store keypoints for CSV export
        setRealTimeFrames(prevFrames => [...prevFrames, response.data.keypoints]);
      }
    } catch (err) {
      console.error("Real-time detection error:", err);
      
      // After errors, try the fallback endpoint
      try {
        const fallbackResponse = await axios.post(`${BACKEND_URL}/detect_pose_simple`, {
          image: imageData,
          save_csv: false
        }, {
          headers: {
            'Content-Type': 'application/json'
          },
          timeout: 3000
        });
        
        if (fallbackResponse.data && fallbackResponse.data.keypoints) {
          setRealTimeKeypoints(fallbackResponse.data.keypoints);
          setRealTimeFrames(prevFrames => [...prevFrames, fallbackResponse.data.keypoints]);
        }
      } catch (fallbackErr) {
        console.error("Fallback real-time detection failed:", fallbackErr);
      }
    }
  };
  
  // Add function to save all real-time frames to a single CSV when tracking is stopped
  const saveRealTimeFramesToCSV = async () => {
    if (realTimeFrames.length === 0) return;
    
    try {
      setLoading(true);
      
      const response = await axios.post(`${BACKEND_URL}/save_realtime_frames`, {
        frames: realTimeFrames
      }, {
        headers: {
          'Content-Type': 'application/json'
        },
        timeout: 30000
      });
      
      setDetectionResults(response.data);
      setRealTimeFrames([]); // Clear frames after saving
      
    } catch (err) {
      console.error("Error saving real-time frames:", err);
      setError("Failed to save real-time tracking data. Please try again.");
    } finally {
      setLoading(false);
    }
  };
  
  const captureImage = () => {
    if (!videoRef.current) return;
    
    try {
      setError(null);
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      
      // Set canvas dimensions to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // Draw video frame to canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Convert canvas to base64 image
      const imageData = canvas.toDataURL('image/jpeg', 0.9);
      
      // Send the image to the backend
      sendImageToBackend(imageData);
    } catch (err) {
      console.error("Error capturing image:", err);
      setError("Error capturing image. Please try again.");
    }
  };
  
  // Original image sending function for manual captures
  const sendImageToBackend = async (imageData) => {
    setLoading(true);
    setError(null);
    try {
      // Log image data size for debugging
      console.log(`Sending image data of length: ${imageData.length}`);
      
      // Try the main endpoint first
      try {
        const response = await axios.post(`${BACKEND_URL}/detect_pose`, {
          image: imageData
        }, {
          headers: {
            'Content-Type': 'application/json'
          },
          timeout: 30000 // 30 second timeout
        });
        
        console.log("Response received:", response.data);
        setDetectionResults(response.data);
        
        // Draw keypoints on canvas if results are available
        if (response.data.keypoints) {
          drawKeypoints(response.data.keypoints);
        } else {
          setError("No keypoints were detected in the image.");
        }
      } catch (primaryError) {
        console.warn("Primary endpoint failed, trying fallback:", primaryError);
        
        // If the primary endpoint fails, try the simple fallback
        const fallbackResponse = await axios.post(`${BACKEND_URL}/detect_pose_simple`, {
          image: imageData
        }, {
          headers: {
            'Content-Type': 'application/json'
          },
          timeout: 30000
        });
        
        console.log("Fallback response received:", fallbackResponse.data);
        setDetectionResults(fallbackResponse.data);
        
        if (fallbackResponse.data.keypoints) {
          drawKeypoints(fallbackResponse.data.keypoints);
        } else {
          setError("No keypoints were detected using the fallback method.");
        }
      }
    } catch (err) {
      console.error("Error sending image to backend:", err);
      if (err.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        console.error("Response data:", err.response.data);
        console.error("Response status:", err.response.status);
        setError(`Server error: ${err.response.status}. ${err.response.data.error || 'Unknown error'}`);
      } else if (err.request) {
        // The request was made but no response was received
        setError("No response from server. Please check if the backend is running.");
      } else {
        // Something happened in setting up the request that triggered an Error
        setError(`Error: ${err.message}`);
      }
    } finally {
      setLoading(false);
    }
  };
  
  // Modify the drawKeypoints function to accept a parameter that controls redrawing the video
  const drawKeypoints = (keypoints, redrawVideo = true) => {
    try {
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      
      // Redraw the video frame only if requested
      // This prevents flickering in real-time mode
      if (redrawVideo) {
        const video = mode === 'webcam' ? videoRef.current : uploadVideoRef.current;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
      }
      
      // Define connections between keypoints for skeleton
      const connections = [
        // Face
        ['nose', 'left_eye'], ['nose', 'right_eye'],
        ['left_eye', 'left_ear'], ['right_eye', 'right_ear'],
        
        // Upper body
        ['left_shoulder', 'right_shoulder'],
        ['left_shoulder', 'left_elbow'], ['right_shoulder', 'right_elbow'],
        ['left_elbow', 'left_wrist'], ['right_elbow', 'right_wrist'],
        
        // Torso
        ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
        ['left_hip', 'right_hip'],
        
        // Lower body
        ['left_hip', 'left_knee'], ['right_hip', 'right_knee'],
        ['left_knee', 'left_ankle'], ['right_knee', 'right_ankle']
      ];
      
      // Convert keypoints array to a map for easier lookup
      const keypointMap = {};
      keypoints.forEach(keypoint => {
        keypointMap[keypoint.name] = {
          x: keypoint.x * canvas.width,
          y: keypoint.y * canvas.height,
          score: keypoint.score
        };
      });
      
      // Draw the connections (skeleton)
      context.lineWidth = 4;
      context.strokeStyle = 'lime';
      
      connections.forEach(([startPoint, endPoint]) => {
        const start = keypointMap[startPoint];
        const end = keypointMap[endPoint];
        
        // Only draw if both points have good confidence scores
        if (start && end && start.score > 0.3 && end.score > 0.3) {
          context.beginPath();
          context.moveTo(start.x, start.y);
          context.lineTo(end.x, end.y);
          context.stroke();
        }
      });
      
      // Draw each keypoint
      keypoints.forEach(keypoint => {
        const { y, x, score, name } = keypoint;
        const pointX = x * canvas.width;
        const pointY = y * canvas.height;
        
        // Only draw keypoints with a confidence score above a threshold
        if (score > 0.3) {
          // Draw the point
          context.beginPath();
          context.arc(pointX, pointY, 6, 0, 2 * Math.PI);
          context.fillStyle = 'red';
          context.fill();
          context.lineWidth = 2;
          context.strokeStyle = 'white';
          context.stroke();
        }
      });
    } catch (err) {
      console.error("Error drawing keypoints:", err);
    }
  };
  
  // Modify the toggle function for real-time tracking
  const toggleRealTimeTracking = () => {
    if (realTimeDetection) {
      // If turning off real-time tracking, save the frames
      setRealTimeDetection(false);
      setRealTimeKeypoints(null);
      
      // Save all collected frames to CSV
      if (realTimeFrames.length > 0) {
        saveRealTimeFramesToCSV();
      }
    } else {
      // If turning on real-time tracking, clear previous frames
      setRealTimeFrames([]);
      setRealTimeDetection(true);
    }
  };
  
  const toggleCapture = () => {
    if (isCapturing) {
      clearInterval(window.captureInterval);
      setIsCapturing(false);
    } else {
      // Capture every 2 seconds
      captureImage(); // Capture one frame immediately
      window.captureInterval = setInterval(captureImage, 2000);
      setIsCapturing(true);
    }
  };
  
  const captureSingleImage = () => {
    captureImage();
  };
  
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    setError(null);
    
    // Create a URL for the uploaded video
    const videoURL = URL.createObjectURL(file);
    
    // Set video details
    setVideoFileDetails({
      name: file.name,
      size: (file.size / (1024 * 1024)).toFixed(2) + ' MB',
      type: file.type
    });
    
    // Set the uploaded video
    setUploadedVideo(videoURL);
    
    // Reset results
    setDetectionResults(null);
    
    // Load the video
    if (uploadVideoRef.current) {
      uploadVideoRef.current.src = videoURL;
      uploadVideoRef.current.onloadedmetadata = () => {
        // Initialize canvas with first frame when video is ready
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        canvas.width = uploadVideoRef.current.videoWidth;
        canvas.height = uploadVideoRef.current.videoHeight;
        context.drawImage(uploadVideoRef.current, 0, 0, canvas.width, canvas.height);
      };
    }
  };
  
  const captureFromUploadedVideo = () => {
    if (!uploadVideoRef.current) return;
    
    try {
      setError(null);
      const video = uploadVideoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      
      // Set canvas dimensions to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // Draw video frame to canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Convert canvas to base64 image
      const imageData = canvas.toDataURL('image/jpeg', 0.9);
      
      // Send the image to the backend
      sendImageToBackend(imageData);
    } catch (err) {
      console.error("Error capturing from video:", err);
      setError("Error capturing frame from video. Please try again.");
    }
  };
  
  const processEntireVideo = async () => {
    if (!uploadVideoRef.current || !fileInputRef.current.files[0]) return;
    
    setProcessingVideo(true);
    setProcessProgress(0);
    setError(null);
    
    try {
      // Create a form data object to send the video file
      const formData = new FormData();
      formData.append('video', fileInputRef.current.files[0]);
      
      // Send the video to the server for processing
      const response = await axios.post(`${BACKEND_URL}/process_video`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setProcessProgress(percentCompleted);
        },
        timeout: 300000 // 5 minute timeout for large videos
      });
      
      setDetectionResults(response.data);
    } catch (err) {
      console.error("Error processing video:", err);
      if (err.response) {
        setError(`Server error: ${err.response.status}. ${err.response.data.error || 'Unknown error'}`);
      } else if (err.request) {
        setError("No response from server. The video might be too large or the processing timed out.");
      } else {
        setError(`Error: ${err.message}`);
      }
    } finally {
      setProcessingVideo(false);
    }
  };
  
  const switchMode = (newMode) => {
    // Clear previous results
    setDetectionResults(null);
    setError(null);
    
    // Stop continuous capture if active
    if (isCapturing) {
      clearInterval(window.captureInterval);
      setIsCapturing(false);
    }
    
    // Switch mode
    setMode(newMode);
  };
  
  return (
    <div className="pose-detection-container">
      <h2>Pose Detection with MoveNet</h2>
      
      <div className="mode-selector">
        <button 
          onClick={() => switchMode('webcam')} 
          className={mode === 'webcam' ? 'active' : ''}
        >
          Webcam Mode
        </button>
        <button 
          onClick={() => switchMode('upload')} 
          className={mode === 'upload' ? 'active' : ''}
        >
          Upload Video Mode
        </button>
      </div>
      
      {error && (
        <div className="error-message">
          {error}
        </div>
      )}
      
      <div className="video-container">
        {mode === 'webcam' && (
          <video 
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{ display: 'none' }}
          />
        )}
        
        {mode === 'upload' && (
          <>
            <div className="upload-controls">
              <input 
                type="file" 
                accept="video/*" 
                onChange={handleFileUpload} 
                ref={fileInputRef}
              />
              
              {uploadedVideo && (
                <video 
                  ref={uploadVideoRef}
                  controls
                  playsInline
                  style={{ maxWidth: '100%', marginBottom: '10px' }}
                />
              )}
            </div>
          </>
        )}
        
        <canvas 
          ref={canvasRef}
          style={{ maxWidth: '100%', border: '1px solid #ccc' }}
        />
      </div>
      
      <div className="controls">
        {mode === 'webcam' && (
          <>
            <button 
              onClick={toggleRealTimeTracking} 
              className={realTimeDetection ? 'active' : ''}
            >
              {realTimeDetection ? 'Stop Real-time Tracking' : 'Enable Real-time Tracking'}
            </button>
            <button onClick={captureSingleImage} disabled={loading}>
              {loading ? 'Processing...' : 'Capture Single Frame'}
            </button>
            <button onClick={toggleCapture} disabled={loading}>
              {isCapturing ? 'Stop Continuous Capture' : 'Start Continuous Capture'}
            </button>
          </>
        )}
        
        {mode === 'upload' && uploadedVideo && (
          <>
            <button onClick={captureFromUploadedVideo} disabled={loading || processingVideo}>
              Capture Current Frame
            </button>
            <button onClick={processEntireVideo} disabled={loading || processingVideo}>
              {processingVideo ? `Processing (${processProgress}%)` : 'Process Entire Video'}
            </button>
          </>
        )}
      </div>
      
      {mode === 'upload' && videoFileDetails && (
        <div className="file-details">
          <h4>Video Details:</h4>
          <p>Name: {videoFileDetails.name}</p>
          <p>Size: {videoFileDetails.size}</p>
          <p>Type: {videoFileDetails.type}</p>
        </div>
      )}
      
      {(loading || processingVideo) && (
        <div className="processing-indicator">
          {loading ? (
            <p>Processing frame...</p>
          ) : (
            <>
              <p>Processing video: {processProgress}%</p>
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ width: `${processProgress}%` }}
                ></div>
              </div>
            </>
          )}
        </div>
      )}
      
      {detectionResults && (
        <div className="results">
          <h3>Detection Results</h3>
          {detectionResults.csv_filename && (
            <p>Saved to CSV: {detectionResults.csv_filename}</p>
          )}
          {detectionResults.json_filename && (
            <p>Saved to JSON: {detectionResults.json_filename}</p>
          )}
          {detectionResults.frames_processed && (
            <p>Frames processed: {detectionResults.frames_processed}</p>
          )}
          {detectionResults.keypoints && (
            <p>Number of keypoints detected: {detectionResults.keypoints.length}</p>
          )}
        </div>
      )}
    </div>
  );
};

export default PoseDetection;