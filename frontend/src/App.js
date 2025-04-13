import React, { useEffect, useState } from 'react';
import axios from 'axios';
import CssBaseline from "@mui/material/CssBaseline";
import Container from "@mui/material/Container";
import Navbar from "./components/Navbar";
import TabView from "./components/TabView"; 
import PoseDetection from "./components/poseDetection"; 

function App() {
  const [message, setMessage] = useState('');

  useEffect(() => {
    axios.get('http://127.0.0.1:5001/api/hello')
      .then(response => setMessage(response.data.message))
      .catch(error => console.error('Error fetching data:', error));
  }, []);

  return (
    <>
      <PoseDetection/>
    </>
  );
}

export default App;
