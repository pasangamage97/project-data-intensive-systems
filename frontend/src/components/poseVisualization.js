import React, { useState } from 'react';
import './poseVisualization.css';

const PoseVisualization = ({ keypoints }) => {
  const [selectedJoint, setSelectedJoint] = useState(null);

  // Joint colors by body part - kept for future 3D visualization
  const jointColors = {
    head: 0xffff00, // yellow
    shoulder: 0x00ffff, // cyan
    elbow: 0xff00ff, // magenta
    wrist: 0xff0000, // red
    hip: 0x00ff00, // green
    knee: 0x0000ff, // blue
    ankle: 0xff8800, // orange
    default: 0xffffff // white
  };

  // Select a joint to highlight in the table
  const handleJointClick = (jointName) => {
    setSelectedJoint(selectedJoint === jointName ? null : jointName);
  };
  
  return (
    <div className="pose-visualization-container">
      <div className="visualization-controls">
        <div className="legend">
          {Object.entries(jointColors).map(([type, color]) => (
            <div key={type} className="legend-item">
              <span className="color-box" style={{ backgroundColor: `#${color.toString(16).padStart(6, '0')}` }}></span>
              <span>{type}</span>
            </div>
          ))}
        </div>
      </div>
      
      <div className="coordinates-panel">
        <h3>Joint Coordinates (x, y, z)</h3>
        <div className="coordinates-scroll">
          <table>
            <thead>
              <tr>
                <th>Joint</th>
                <th>X</th>
                <th>Y</th>
                <th>Z</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {keypoints.map((keypoint, index) => (
                <tr 
                  key={index} 
                  className={selectedJoint === keypoint.name ? 'selected-joint' : ''}
                  onClick={() => handleJointClick(keypoint.name)}
                >
                  <td>{keypoint.name}</td>
                  <td>{keypoint.x.toFixed(4)}</td>
                  <td>{keypoint.y.toFixed(4)}</td>
                  <td>{keypoint.z ? keypoint.z.toFixed(4) : 'N/A'}</td>
                  <td>{keypoint.score.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default PoseVisualization;