import React, { useState, useEffect } from "react";
import { Box, Button, FormControl, InputLabel, MenuItem, Select, Typography, Alert, CircularProgress } from "@mui/material";

export default function FormComponent({ title, dropdownOptions, submitEndpoint, systemStatusEndpoint }) {
  const [dropdownValue, setDropdownValue] = useState("");
  const [file, setFile] = useState(null);
  const [alertMessage, setAlertMessage] = useState("");
  const [alertSeverity, setAlertSeverity] = useState("info");
  const [isSubmitting, setIsSubmitting] = useState(false); // Submission state

  useEffect(() => {
    const checkSystemStatus = async () => {
      try {
        const response = await fetch(systemStatusEndpoint, { method: "GET" });

        if (response.ok) {
          setAlertMessage("System is working fine!");
          setAlertSeverity("success");
        } else {
          setAlertMessage("System issue detected!");
          setAlertSeverity("error");
        }
      } catch (error) {
        setAlertMessage("Error: Unable to reach the server.");
        setAlertSeverity("error");
      }
    };

    checkSystemStatus();
  }, [systemStatusEndpoint]);

  const handleDropdownChange = (event) => {
    setDropdownValue(event.target.value);
  };

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    // Validate if a model is selected and a CSV file is uploaded
    if (!dropdownValue) {
      setAlertMessage("Please select a model.");
      setAlertSeverity("error");
      return;
    }

    if (!file || file.type !== "text/csv") {
      setAlertMessage("Please upload a valid CSV file.");
      setAlertSeverity("error");
      return;
    }

    setIsSubmitting(true); // Start loading state

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model", dropdownValue);

    try {
      const response = await fetch(submitEndpoint, {
        method: "POST",
        body: formData, // Use FormData for file uploads
      });

      if (response.ok) {
        const result = await response.json();
        setAlertMessage(result.message || "Form submitted successfully!");
        setAlertSeverity("success");
      } else {
        const errorResult = await response.json();
        setAlertMessage(errorResult.error || "Submission failed!");
        setAlertSeverity("error");
      }
    } catch (error) {
      setAlertMessage("Error during form submission.");
      setAlertSeverity("error");
    } finally {
      setIsSubmitting(false); // Stop loading state
    }
  };

  return (
    <Box 
      component="form" 
      onSubmit={handleSubmit} 
      sx={{ p: 3, border: "1px solid #ccc", borderRadius: 2, mt: 2, maxWidth: "100%", width: "100%", boxSizing: "border-box" }}
    >
      {/* Always visible alert box for system status */}
      {alertMessage && (
        <Alert severity={alertSeverity} sx={{ mb: 2 }}>
          {alertMessage}
        </Alert>
      )}

      {/* Title */}
      <Typography variant="h5" sx={{ mb: 2, textAlign: "left" }}>
        {title}
      </Typography>

      {/* Dropdown List */}
      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>Select a model</InputLabel>
        <Select value={dropdownValue} onChange={handleDropdownChange}>
          {dropdownOptions.map((option, index) => (
            <MenuItem key={index} value={option.value}>
              {option.label}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {/* File Input - Custom Button */}
      <Button 
        variant="outlined" 
        component="label" 
        sx={{ mb: 2, width: "100%" }}
      >
        {file ? file.name : "Choose a CSV file"}
        <input type="file" hidden onChange={handleFileChange} accept=".csv" />
      </Button>

      {/* Submit Button - Full width, disabled and loading state */}
      <Button 
        type="submit" 
        variant="contained" 
        sx={{ width: "100%" }}
        disabled={isSubmitting}
      >
        {isSubmitting ? <CircularProgress size={24} /> : "Submit"} {/* Show loading spinner */}
      </Button>
    </Box>
  );
}
