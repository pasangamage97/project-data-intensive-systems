import React, { useState } from "react";
import { Tabs, Tab, Box, Container } from "@mui/material";
import FormComponent from "./FormComponent"; // ✅ Import FormComponent

export default function TabView() {
  const [activeTab, setActiveTab] = useState(0);

  // Define dropdown options for each tab
  const dropdownOptionsTabOne = [
    { value: "option1", label: "Option 1" },
    { value: "option2", label: "Option 2" },
    { value: "option3", label: "Option 3" },
  ];

  const dropdownOptionsTabTwo = [
    { value: "optionA", label: "Option A" },
    { value: "optionB", label: "Option B" },
    { value: "optionC", label: "Option C" },
  ];

  // Define submit endpoints for each tab
  const submitEndpointTabOne = "https://api.example.com/submitTabOne";
  const submitEndpointTabTwo = "https://api.example.com/submitTabTwo";

  return (
    <Container maxWidth="md">
      {/* Tabs Navigation */}
      <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)}>
        <Tab label="Regression" />
        <Tab label="Classification" />
      </Tabs>

      {/* Tab Content - Left aligned */}
      <Box sx={{ mt: 2, p: 2, pl:0, display: "flex", justifyContent: "flex-start" }}>
        {activeTab === 0 && (
          <FormComponent
            title="Make Prediction"
            dropdownOptions={dropdownOptionsTabOne}
            submitEndpoint={submitEndpointTabOne}
          />
        )}
        {activeTab === 1 && (
          <FormComponent
            title="Select Model & Classify"
            dropdownOptions={dropdownOptionsTabTwo}
            submitEndpoint={submitEndpointTabTwo}
          />
        )}
      </Box>
    </Container>
  );
}
