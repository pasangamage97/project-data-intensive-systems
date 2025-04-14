// App.js
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import CssBaseline from "@mui/material/CssBaseline";
import Container from "@mui/material/Container";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Paper from "@mui/material/Paper";
import Navbar from "./components/Navbar";
import TabView from "./components/TabView"; 
import PoseDetection from "./components/poseDetection"; // Make sure the case matches your file

// Add some basic styling for PoseDetection
const styles = {
  container: {
    marginTop: '24px',
    marginBottom: '24px',
  },
  paper: {
    padding: '24px',
    borderRadius: '8px',
  },
  message: {
    marginTop: '16px',
    padding: '12px',
    borderRadius: '4px',
    backgroundColor: '#f5f5f5',
  }
};

function App() {
  const [message, setMessage] = useState('');
  const [apiStatus, setApiStatus] = useState('loading');
  
  useEffect(() => {
    // Check backend health
    axios.get('http://127.0.0.1:5001/health')
      .then(response => {
        if (response.data.status === 'ok') {
          setApiStatus('connected');
        } else {
          setApiStatus('error');
        }
      })
      .catch(error => {
        console.error('Backend connection error:', error);
        setApiStatus('error');
      });
      
    // Get hello message
    axios.get('http://127.0.0.1:5001/api/hello')
      .then(response => setMessage(response.data.message))
      .catch(error => console.error('Error fetching data:', error));
  }, []);

  return (
    <>
      <CssBaseline />
      <Navbar />
      <Container maxWidth="lg" style={styles.container}>
        {apiStatus === 'error' && (
          <Paper 
            elevation={3} 
            style={{...styles.paper, backgroundColor: '#ffe0e0', marginBottom: '20px'}}
          >
            <Typography variant="h6" color="error">
              Backend Connection Error
            </Typography>
            <Typography variant="body1">
              Unable to connect to the backend server. Please make sure your Flask server is running on port 5001.
            </Typography>
          </Paper>
        )}
        
        <Paper elevation={3} style={styles.paper}>
          <Typography variant="h4" gutterBottom align="center">
            MoveNet Pose Detection
          </Typography>
          
          <PoseDetection />
          
          {message && (
            <Box style={styles.message}>
              <Typography variant="body2" color="textSecondary">
                {message}
              </Typography>
            </Box>
          )}
        </Paper>
      </Container>
    </>
  );
}

export default App;