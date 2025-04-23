// PoseDetection.js - Enhanced for dual 2D and 3D visualization
import React, { useRef, useState, useEffect } from 'react';
import axios from 'axios';
import './PoseDetection.css';
import PoseVisualization from './poseVisualization';
import Pose3DVisualization from './Pose3DVisualization'; // Import the 3D component

// Define the backend URL as a constant
const BACKEND_URL = 'http://127.0.0.1:5001';

const PoseDetection = () => {
  // References
  const videoRef = useRef(null);
  const uploadVideoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const processedVideoRef = useRef(null);
  
  // State variables
  const [isCapturing, setIsCapturing] = useState(false);
  const [detectionResults, setDetectionResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState('webcam'); // 'webcam' or 'upload'
  const [uploadedVideo, setUploadedVideo] = useState(null);
  const [processedVideo, setProcessedVideo] = useState(null);
  const [processingVideo, setProcessingVideo] = useState(false);
  const [processProgress, setProcessProgress] = useState(0);
  const [videoFileDetails, setVideoFileDetails] = useState(null);
  const [error, setError] = useState(null);
  const [realTimeDetection, setRealTimeDetection] = useState(false);
  const [realTimeKeypoints, setRealTimeKeypoints] = useState(null);
  const [realTimeFrames, setRealTimeFrames] = useState([]);
  const [showCoordinates, setShowCoordinates] = useState(false);
  const [capturedImageData, setCapturedImageData] = useState(null);
  const [showCapturedImage, setShowCapturedImage] = useState(false);
  const [currentKeypoints, setCurrentKeypoints] = useState(null);
  
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
  
  // Animation loop for real-time detection in webcam mode
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
  }, [mode, realTimeDetection, realTimeKeypoints]);
  
  // Add a video time update handler for processed video playback
  useEffect(() => {
    const processedVideo = processedVideoRef.current;
    
    if (!processedVideo) return;
    
    const handleTimeUpdate = () => {
      if (detectionResults && detectionResults.json_filename) {
        // This is a placeholder for real-time 3D visualization during video playback
        // In a full implementation, you would load keypoints corresponding to the current video time
        // For now, we'll leave the current keypoints unchanged
      }
    };
    
    processedVideo.addEventListener('timeupdate', handleTimeUpdate);
    
    return () => {
      if (processedVideo) {
        processedVideo.removeEventListener('timeupdate', handleTimeUpdate);
      }
    };
  }, [detectionResults, processedVideoRef.current]);
  
  // Function to start webcam
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
  
  // Function to clear the canvas
  const clearCanvas = () => {
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      
      // Check if we need to redraw the video frame
      if (mode === 'webcam' && videoRef.current) {
        // For webcam mode, redraw the current video frame without keypoints
        context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
      } else if (mode === 'upload' && uploadVideoRef.current) {
        // For upload mode, redraw the current video frame without keypoints
        context.drawImage(uploadVideoRef.current, 0, 0, canvas.width, canvas.height);
      } else {
        // If no video frame to redraw, clear the canvas completely
        context.clearRect(0, 0, canvas.width, canvas.height);
      }
    }
  };
  
  // Function to send image to backend for real-time tracking
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
        setCurrentKeypoints(response.data.keypoints);
        
        // Store keypoints for CSV export
        setRealTimeFrames(prevFrames => [...prevFrames, response.data.keypoints]);
      }
    } catch (err) {
      console.error("Real-time detection error:", err);
      
      // REPLACE THIS SECTION with the new fallback code
      try {
        const fallbackResponse = await axios.post(`${BACKEND_URL}/detect_pose_simple`, {
          image: "placeholder", // Just send a string instead of image data
          save_csv: false
        }, {
          headers: {
            'Content-Type': 'application/json'
          },
          timeout: 3000
        });
        
        if (fallbackResponse.data.keypoints) {
          setRealTimeKeypoints(fallbackResponse.data.keypoints);
          setCurrentKeypoints(fallbackResponse.data.keypoints);
          setRealTimeFrames(prevFrames => [...prevFrames, fallbackResponse.data.keypoints]);
        }
      } catch (fallbackErr) {
        console.error("Fallback real-time detection failed:", fallbackErr);
      }
    }
  };
  
  // Function to save real-time frames to CSV
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
  
  // Function to capture a single image from webcam
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
  
  // Function to send image to backend and process the response
  const sendImageToBackend = async (imageData) => {
    setLoading(true);
    setError(null);
    setCapturedImageData(imageData);
  
    try {
      console.log("Sending image to backend...");
      const response = await axios.post(`${BACKEND_URL}/detect_pose`, {
        image: imageData,
        save_csv: true
      }, {
        headers: {
          'Content-Type': 'application/json'
        },
        timeout: 30000
      });
  
      // Rest of the existing function...
  
    } catch (err) {
      console.error("Error sending image to backend:", err);
      
      // REPLACE THIS SECTION with the new fallback code
      try {
        console.log("Trying fallback endpoint...");
        // Use an empty image or very small placeholder to avoid decode issues
        const fallbackResponse = await axios.post(`${BACKEND_URL}/detect_pose_simple`, {
          image: "placeholder",  // Just send a string instead of image data
          save_csv: true
        }, {
          headers: {
            'Content-Type': 'application/json'
          },
          timeout: 30000
        });
        
        if (fallbackResponse.data.keypoints) {
          drawKeypoints(fallbackResponse.data.keypoints);
          setCurrentKeypoints(fallbackResponse.data.keypoints);
          
          const canvasWithKeypoints = canvasRef.current.toDataURL('image/jpeg', 0.9);
          setCapturedImageData(canvasWithKeypoints);
          setShowCapturedImage(true);
          
          setDetectionResults(fallbackResponse.data);
          setError("Using fallback pose detection (simulated data).");
        }
      } catch (fallbackErr) {
        console.error("Fallback also failed:", fallbackErr);
        setError("Error processing image. Backend service may be unavailable.");
      }
    } finally {
      setLoading(false);
    }
  };
  
  // Function to toggle continuous capture
  const toggleCapture = () => {
    if (isCapturing) {
      clearInterval(window.captureInterval);
      setIsCapturing(false);
      
      // Clear the canvas
      clearCanvas();
    } else {
      // Capture every 2 seconds
      captureImage(); // Capture one frame immediately
      window.captureInterval = setInterval(captureImage, 2000);
      setIsCapturing(true);
    }
  };
  
  // Function to capture a single frame from webcam
  const captureSingleImage = () => {
    // Clear previous image from real-time
    setRealTimeKeypoints(null);
    setShowCapturedImage(false); // reset in case
  
    // Then capture
    captureImage();
    setShowCapturedImage(true);
  };
  
  // Function to handle file upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    setError(null);
    
    // Validate file type
    if (!file.type.startsWith('video/')) {
      setError("Please upload a valid video file.");
      return;
    }
    
    // Reset previous results
    setDetectionResults(null);
    setCurrentKeypoints(null);
    setCapturedImageData(null);
    setShowCapturedImage(false);
    setProcessedVideo(null);
    
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
    
    // Load the video and set up event listeners
    if (uploadVideoRef.current) {
      uploadVideoRef.current.src = videoURL;
      
      // Add event listeners to update canvas when video is ready
      uploadVideoRef.current.onloadedmetadata = updateCanvasForVideo;
      uploadVideoRef.current.onloadeddata = updateCanvasForVideo;
      uploadVideoRef.current.onplay = updateCanvasForVideo;
      uploadVideoRef.current.onseeked = updateCanvasForVideo;
      
      // Ensure canvas updates when video is paused or seeking completes
      uploadVideoRef.current.addEventListener('pause', updateCanvasForVideo);
      uploadVideoRef.current.addEventListener('seeked', updateCanvasForVideo);
    }
  };
  
  // Function to update canvas for video
  const updateCanvasForVideo = () => {
    if (uploadVideoRef.current && canvasRef.current && uploadVideoRef.current.readyState >= 2) {
      const video = uploadVideoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      
      // Adjust canvas size to match video dimensions
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // Clear previous content and draw the current video frame
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      console.log(`Canvas resized to match video: ${canvas.width}x${canvas.height}`);
    }
  };
  
  // Function to capture from uploaded video
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
      
      // Make sure the capture is visible
      setShowCapturedImage(true);
    } catch (err) {
      console.error("Error capturing from video:", err);
      setError("Error capturing frame from video. Please try again.");
    }
  };
  
  // Function to process the entire video
  const processEntireVideo = async () => {
    if (!uploadVideoRef.current || !fileInputRef.current.files[0]) return;
  
    setProcessingVideo(true);
    setProcessProgress(0);
    setError(null);
  
    try {
      const formData = new FormData();
      formData.append('video', fileInputRef.current.files[0]);
  
      const response = await axios.post(`${BACKEND_URL}/process_video`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setProcessProgress(percentCompleted);
        },
        timeout: 300000
      });
  
      if (response.data.processed_video_url) {
        setProcessedVideo(`${BACKEND_URL}${response.data.processed_video_url}`);
      }
  
      setDetectionResults(response.data);
      setCurrentKeypoints(null); // optionally reset keypoints after video
  
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


  const handleBulkUpload = async (event) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;
  
    const formData = new FormData();
    for (let file of files) {
      formData.append("videos", file);
    }
  
    setLoading(true);
    setError(null);
  
    try {
      const response = await axios.post(`${BACKEND_URL}/process_bulk_videos`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 10 * 60 * 1000 // up to 10 minutes for large batches
      });
  
      console.log("Bulk upload response:", response.data);
  
      alert(`Processed ${response.data.results.length} videos! Output folder: ${response.data.output_folder}`);
    } catch (err) {
      console.error("Bulk upload failed:", err);
      setError("Bulk upload failed. Check server or try fewer files.");
    } finally {
      setLoading(false);
    }
  };
  
  // Function to switch modes (webcam/upload)
  const switchMode = (newMode) => {
    // Clear previous results
    setDetectionResults(null);
    setError(null);
    setRealTimeKeypoints(null);
    setCurrentKeypoints(null);
    
    // Stop continuous capture if active
    if (isCapturing) {
      clearInterval(window.captureInterval);
      setIsCapturing(false);
    }
    
    // Stop real-time tracking if active
    if (realTimeDetection) {
      setRealTimeDetection(false);
    }
    
    // Clear the canvas
    setTimeout(clearCanvas, 100); // Small delay to ensure canvas exists
    
    // Switch mode
    setMode(newMode);
  };
  
  // Toggle real-time tracking
  const toggleRealTimeTracking = () => {
    if (realTimeDetection) {
      setRealTimeDetection(false);
      setRealTimeKeypoints(null);
      clearCanvas();
      setShowCapturedImage(false); // also remove captured image
  
      if (realTimeFrames.length > 0) {
        saveRealTimeFramesToCSV();
      }
    } else {
      setRealTimeFrames([]);
      setRealTimeDetection(true);
      setShowCapturedImage(false); // hide the image if coming from capture
    }
  };
  
  // Toggle coordinates visibility
  const toggleCoordinates = () => {
    setShowCoordinates(!showCoordinates);
  };
  
  // Function to draw keypoints on canvas
  const drawKeypoints = (keypoints, redrawVideo = true) => {
    try {
      if (!canvasRef.current) {
        console.error("Canvas reference is null");
        return;
      }
      
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      
      // Log for debugging
      console.log(`Drawing ${keypoints.length} keypoints on canvas`);
      
      // Ensure canvas has proper dimensions
      if (canvas.width === 0 || canvas.height === 0) {
        console.warn("Canvas has zero width or height");
        
        // Try to fix by getting dimensions from video
        const video = mode === 'webcam' ? videoRef.current : uploadVideoRef.current;
        if (video && video.videoWidth && video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          console.log(`Reset canvas dimensions to ${canvas.width}x${canvas.height}`);
        } else {
          // Set default dimensions as fallback
          canvas.width = 640;
          canvas.height = 480;
          console.log("Using default dimensions for canvas");
        }
      }
      
      // Redraw the video frame only if requested
      if (redrawVideo) {
        const video = mode === 'webcam' ? videoRef.current : uploadVideoRef.current;
        if (video && video.readyState >= 2) {
          context.drawImage(video, 0, 0, canvas.width, canvas.height);
        } else {
          // If video isn't ready, just clear the canvas
          context.clearRect(0, 0, canvas.width, canvas.height);
          context.fillStyle = "#000000";
          context.fillRect(0, 0, canvas.width, canvas.height);
        }
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
      
      console.log("Keypoints drawn successfully");
    } catch (err) {
      console.error("Error drawing keypoints:", err);
    }
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
            <button onClick={clearCanvas}>Reset</button>
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

      <div className="bulk-upload-section">
  <h3>Bulk Video Upload (PoseNet CSV Extraction)</h3>
  <input
    type="file"
    accept="video/*"
    multiple
    onChange={handleBulkUpload}
  />
</div>
      
      {/* Processed Video Section (only shows when a video has been processed) */}
      {processedVideo && (
        <div className="processed-video-container">
          <h3>Processed Video with Skeleton</h3>
          <video 
            ref={processedVideoRef}
            controls
            playsInline
            src={processedVideo}
            style={{ maxWidth: '100%', marginBottom: '10px', border: '1px solid #ccc' }}
          />
        </div>
      )}
      
      {/* Visualization Controls */}
      {currentKeypoints && (
        <div className="visualization-controls">
          <button 
            onClick={toggleCoordinates}
            className={showCoordinates ? 'active' : ''}
          >
            {showCoordinates ? 'Hide Coordinates Table' : 'Show Coordinates Table'}
          </button>
        </div>
      )}
      
      {/* Dual visualization container with both 2D and 3D views */}
      {currentKeypoints && (
        <div className="visualizations-container">
          {/* 3D Visualization Component - Always show when keypoints are available */}
          <div className="visualization-pane">
            <Pose3DVisualization 
              keypoints={currentKeypoints} 
              width={400}
              height={300}
            />
          </div>

          {processedVideo && (
          <div className="processed-video-container">
            <h3>Processed 3D Skeleton-Only Video</h3>
            <video
              ref={processedVideoRef}
              controls
              playsInline
              src={processedVideo}
              style={{ maxWidth: '100%', border: '2px solid #444' }}
            />
          </div>
        )}
          
          {/* Coordinates Table (optional visibility) */}
          {showCoordinates && (
            <div className="visualization-pane">
              <div className="visualization-wrapper">
                <h3>Joint Coordinates</h3>
                <PoseVisualization 
                  keypoints={currentKeypoints} 
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Captured Image with Keypoints */}
      {capturedImageData && showCapturedImage && (
        <div className="captured-image-container">
          <h3>Captured Image with Keypoints</h3>
          <img 
            src={capturedImageData} 
            alt="Captured pose" 
            className="captured-image"
          />
        </div>
      )}
      
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