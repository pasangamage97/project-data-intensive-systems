// Pose3DVisualization.js with motion trails
import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import './Pose3DVisualization.css';

const Pose3DVisualization = ({ keypoints, width = 400, height = 300 }) => {
  const containerRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const pointsRef = useRef([]);
  const bonesRef = useRef([]);
  const trailsRef = useRef({});  // Store motion trails for each joint
  const animationIdRef = useRef(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const frameHistoryRef = useRef([]); // Store history of keypoint frames
  const maxTrailLength = 30; // Maximum number of frames to keep in the trail
  
  // Define joint connections for the skeleton
  const skeletonConnections = [
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
  
  // Define colors for different joint types
  const jointColors = {
    nose: 0xffff00,
    eye: 0xffff00,
    ear: 0xffff00,
    shoulder: 0x00ffff,
    elbow: 0xff00ff,
    wrist: 0xff0000,
    hip: 0x00ff00,
    knee: 0x0000ff,
    ankle: 0xff8800,
    default: 0xffffff
  };
  
  // Colors for motion trails
  const trailColors = {
    left_wrist: 0x0000ff,    // Blue for left hand
    right_wrist: 0x0000ff,   // Blue for right hand
    left_ankle: 0x0000ff,    // Blue for left foot
    right_ankle: 0x0000ff,   // Blue for right foot
    left_knee: 0x0000ff,     // Blue for left knee
    right_knee: 0x0000ff,    // Blue for right knee
    head: 0x0000ff,          // Blue for head
    default: 0x0000ff        // Default blue
  };
  
  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current) return;
    
    // Only initialize once
    if (isInitialized) return;
    
    console.log("Initializing 3D visualization component");
    
    try {
      // Create scene
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xf0f0f0); // Light gray background
      sceneRef.current = scene;
      
      // Create camera
      const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
      camera.position.set(0, 0, 2);
      cameraRef.current = camera;
      
      // Create renderer
      const renderer = new THREE.WebGLRenderer({ 
        antialias: true,
        alpha: true 
      });
      renderer.setSize(width, height);
      renderer.setPixelRatio(window.devicePixelRatio);
      
      // Clear any existing canvas
      if (containerRef.current.childNodes.length > 0) {
        containerRef.current.innerHTML = '';
      }
      containerRef.current.appendChild(renderer.domElement);
      rendererRef.current = renderer;
      
      // Add ambient light
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
      scene.add(ambientLight);
      
      // Add directional light
      const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
      directionalLight.position.set(1, 1, 1);
      scene.add(directionalLight);
      
      // Add grid helper for reference
      const gridHelper = new THREE.GridHelper(2, 10, 0x444444, 0x555555);
      scene.add(gridHelper);
      
      // Add axes helper for orientation
      const axesHelper = new THREE.AxesHelper(1);
      scene.add(axesHelper);
      
      // Add orbit controls for interactive viewing
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.25;
      controls.enableZoom = true;
      controlsRef.current = controls;
      
      // Animation loop
      const animate = () => {
        animationIdRef.current = requestAnimationFrame(animate);
        
        // Update controls
        if (controlsRef.current) {
          controlsRef.current.update();
        }
        
        // Render the scene
        if (rendererRef.current && sceneRef.current && cameraRef.current) {
          rendererRef.current.render(sceneRef.current, cameraRef.current);
        }
      };
      
      // Start animation loop
      animate();
      
      // Mark as initialized
      setIsInitialized(true);
      
      console.log("3D visualization component initialized successfully");
    } catch (error) {
      console.error("Error initializing 3D scene:", error);
    }
    
    // Cleanup function
    return () => {
      console.log("Cleaning up 3D visualization");
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
      
      if (rendererRef.current && containerRef.current) {
        containerRef.current.removeChild(rendererRef.current.domElement);
      }
      
      // Clean up references
      pointsRef.current = [];
      bonesRef.current = [];
      trailsRef.current = {};
      frameHistoryRef.current = [];
      sceneRef.current = null;
      cameraRef.current = null;
      rendererRef.current = null;
      controlsRef.current = null;
      
      setIsInitialized(false);
    };
  }, [width, height]);
  
  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      if (cameraRef.current && rendererRef.current && containerRef.current) {
        const newWidth = containerRef.current.clientWidth;
        const newHeight = height;
        
        cameraRef.current.aspect = newWidth / newHeight;
        cameraRef.current.updateProjectionMatrix();
        
        rendererRef.current.setSize(newWidth, newHeight);
      }
    };
    
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [height]);
  
  // Update trails for motion tracking
  const updateTrails = (newKeypoints) => {
    if (!sceneRef.current || !isInitialized) return;
    
    const scene = sceneRef.current;
    
    // Add current keypoints to history
    frameHistoryRef.current.push(newKeypoints);
    
    // Limit the trail length
    if (frameHistoryRef.current.length > maxTrailLength) {
      frameHistoryRef.current.shift();
    }
    
    // Remove old trails
    Object.values(trailsRef.current).forEach(trail => {
      if (trail && scene) {
        scene.remove(trail);
      }
    });
    
    // Reset trails reference
    trailsRef.current = {};
    
    // Important joints to track
    const trackedJoints = [
      'left_wrist', 'right_wrist', 
      'left_ankle', 'right_ankle',
      'left_knee', 'right_knee',
      'nose'
    ];
    
    // Create new trails for each tracked joint
    trackedJoints.forEach(jointName => {
      const positions = [];
      
      // Collect positions from frame history
      frameHistoryRef.current.forEach(frame => {
        const joint = frame.find(kp => kp.name === jointName);
        if (joint && joint.score > 0.3) {
          positions.push(
            new THREE.Vector3(
              (joint.x * 2 - 1) * 0.8,      // Map X from 0-1 to -0.8-0.8
              -(joint.y * 2 - 1) * 0.8,     // Map Y from 0-1 to 0.8--0.8 (inverted)
              joint.z * 0.8                 // Scale Z appropriately
            )
          );
        }
      });
      
      // Only create trail if we have at least 2 positions
      if (positions.length > 1) {
        const geometry = new THREE.BufferGeometry().setFromPoints(positions);
        const material = new THREE.LineBasicMaterial({ 
          color: trailColors[jointName] || trailColors.default,
          linewidth: 2,
          opacity: 0.7,
          transparent: true
        });
        
        const trail = new THREE.Line(geometry, material);
        scene.add(trail);
        trailsRef.current[jointName] = trail;
      }
    });
  };
  
  // Update points and bones when keypoints change
  useEffect(() => {
    if (!sceneRef.current || !isInitialized || !keypoints || keypoints.length === 0) return;
    
    console.log("Updating 3D visualization with new keypoints");
    
    try {
      const scene = sceneRef.current;
      
      // Remove old points and bones
      pointsRef.current.forEach(point => scene.remove(point));
      bonesRef.current.forEach(bone => scene.remove(bone));
      pointsRef.current = [];
      bonesRef.current = [];
      
      // Update trails with new keypoints
      updateTrails(keypoints);
      
      // Create a map for quick lookup of keypoints
      const keypointMap = {};
      keypoints.forEach(keypoint => {
        keypointMap[keypoint.name] = keypoint;
      });
      
      // Create new points for each keypoint
      keypoints.forEach(keypoint => {
        if (keypoint.score < 0.2) return; // Skip very low confidence points
        
        // Extract type from name for coloring
        let type = keypoint.name;
        if (type.includes('_')) {
          type = type.split('_')[1]; // Get general type (eye, shoulder, etc.)
        }
        
        const color = jointColors[type] || jointColors.default;
        
        // Create sphere for joint
        const geometry = new THREE.SphereGeometry(0.03, 16, 16);
        const material = new THREE.MeshStandardMaterial({ 
          color,
          emissive: color,
          emissiveIntensity: 0.3
        });
        const sphere = new THREE.Mesh(geometry, material);
        
        // Get Z value, default to 0 if not present
        const zValue = keypoint.z !== undefined ? keypoint.z : 0;
        
        // Position the point in 3D space
        sphere.position.set(
          (keypoint.x * 2 - 1) * 0.8,      // Map X from 0-1 to -0.8-0.8
          -(keypoint.y * 2 - 1) * 0.8,     // Map Y from 0-1 to 0.8--0.8 (inverted)
          zValue * 0.8                      // Scale Z appropriately
        );
        
        scene.add(sphere);
        pointsRef.current.push(sphere);
        
        // Store sphere reference with keypoint name for connecting bones
        keypointMap[keypoint.name].sphere = sphere;
      });
      
      // Create bones connecting the joints
      skeletonConnections.forEach(([startName, endName]) => {
        const startPoint = keypointMap[startName];
        const endPoint = keypointMap[endName];
        
        // Skip if either point is missing or has low confidence
        if (!startPoint || !endPoint || !startPoint.sphere || !endPoint.sphere || 
            startPoint.score < 0.3 || endPoint.score < 0.3) {
          return;
        }
        
        // Create a line for the bone
        const points = [
          startPoint.sphere.position,
          endPoint.sphere.position
        ];
        
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({ 
          color: 0x00ff00, 
          linewidth: 3
        });
        
        const line = new THREE.Line(geometry, material);
        scene.add(line);
        bonesRef.current.push(line);
      });
      
      console.log(`Created ${pointsRef.current.length} points and ${bonesRef.current.length} bones`);
    } catch (error) {
      console.error("Error updating 3D visualization:", error);
    }
  }, [keypoints, isInitialized]);
  
  return (
    <div className="pose-3d-container">
      <h3>3D Skeleton Visualization with Motion Trails</h3>
      <div 
        ref={containerRef} 
        className="pose-3d-canvas"
        style={{ width: '100%', height: `${height}px` }}
      ></div>
      <div className="pose-3d-info">
        <p>This visualization shows the skeleton with depth information and blue motion trails.</p>
        <p>Use mouse to rotate view, scroll to zoom in/out.</p>
      </div>
    </div>
  );
};

export default Pose3DVisualization;