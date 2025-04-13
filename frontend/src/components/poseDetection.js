// PoseDetection.jsx
import React, { useRef, useState, useEffect } from 'react';
import axios from 'axios';

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
  
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error("Error accessing webcam:", err);
    }
  };
  
  const captureImage = () => {
    if (!videoRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert canvas to base64 image
    const imageData = canvas.toDataURL('image/jpeg');
    
    // Send the image to the backend
    sendImageToBackend(imageData);
  };
  
  const sendImageToBackend = async (imageData) => {
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:5000/detect_pose', {
        image: imageData
      });
      
      setDetectionResults(response.data);
      
      // Draw keypoints on canvas if results are available
      if (response.data.keypoints) {
        drawKeypoints(response.data.keypoints);
      }
    } catch (err) {
      console.error("Error sending image to backend:", err);
    } finally {
      setLoading(false);
    }
  };
  
  const drawKeypoints = (keypoints) => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    
    // Clear previous drawings
    context.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw the video frame first
    if (mode === 'webcam') {
      context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    } else if (mode === 'upload' && uploadVideoRef.current) {
      context.drawImage(uploadVideoRef.current, 0, 0, canvas.width, canvas.height);
    }
    
    // Draw each keypoint
    keypoints.forEach(keypoint => {
      const { y, x, score, name } = keypoint;
      
      // Only draw keypoints with a confidence score above a threshold
      if (score > 0.3) {
        // Draw the point
        context.beginPath();
        context.arc(x * canvas.width, y * canvas.height, 5, 0, 2 * Math.PI);
        context.fillStyle = 'red';
        context.fill();
        
        // Optional: Draw the name of the keypoint
        context.font = '12px Arial';
        context.fillStyle = 'white';
        context.fillText(name, x * canvas.width + 10, y * canvas.height);
      }
    });
  };
  
  const toggleCapture = () => {
    if (isCapturing) {
      clearInterval(window.captureInterval);
      setIsCapturing(false);
    } else {
      // Capture every 1 second
      window.captureInterval = setInterval(captureImage, 1000);
      setIsCapturing(true);
    }
  };
  
  const captureSingleImage = () => {
    captureImage();
  };
  
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
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
    
    const video = uploadVideoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert canvas to base64 image
    const imageData = canvas.toDataURL('image/jpeg');
    
    // Send the image to the backend
    sendImageToBackend(imageData);
  };
  
  const processEntireVideo = async () => {
    if (!uploadVideoRef.current) return;
    
    const video = uploadVideoRef.current;
    setProcessingVideo(true);
    setProcessProgress(0);
    
    try {
      // Create a form data object to send the video file
      const formData = new FormData();
      formData.append('video', fileInputRef.current.files[0]);
      
      // Send the video to the server for processing
      const response = await axios.post('http://localhost:5000/process_video', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setProcessProgress(percentCompleted);
        }
      });
      
      setDetectionResults(response.data);
    } catch (err) {
      console.error("Error processing video:", err);
    } finally {
      setProcessingVideo(false);
    }
  };
  
  const switchMode = (newMode) => {
    // Clear previous results
    setDetectionResults(null);
    
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
            <button onClick={captureSingleImage} disabled={loading}>
              Capture Single Frame
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
              Process Entire Video
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