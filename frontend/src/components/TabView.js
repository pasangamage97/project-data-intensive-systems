import React, { useState, useEffect } from "react";
import { Tabs, Tab, Box } from "@mui/material";
import FormComponent from "./FormComponent";

export default function TabView() {
  const [tabValue, setTabValue] = useState(0);
  const [models, setModels] = useState([]);
  const [categorizingModels, setCategorizingModels] = useState([]);

  // Fetch dropdown data when component mounts
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5001/api/models");
        if (response.ok) {
          const data = await response.json();
          setModels(data);
        }
      } catch (error) {
        console.error("Error fetching models:", error);
      }
    };

    const fetchCategorizingModels = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5001/api/categorizingModels");
        if (response.ok) {
          const data = await response.json();
          setCategorizingModels(data);
        }
      } catch (error) {
        console.error("Error fetching categorizing models:", error);
      }
    };

    fetchModels();
    fetchCategorizingModels();
  }, []);

  return (
    <Box sx={{ width: "100%" }}>
      <Tabs value={tabValue} onChange={(event, newValue) => setTabValue(newValue)}>
        <Tab label="Models" />
        <Tab label="Categorizing Models" />
      </Tabs>

      <Box sx={{ p: 2 }}>
        {tabValue === 0 && (
          <FormComponent
            title="Select a Model"
            dropdownOptions={models}
            submitEndpoint="http://127.0.0.1:5001/api/predict"
            systemStatusEndpoint="/api/system-status"
          />
        )}
        {tabValue === 1 && (
          <FormComponent
            title="Select a Categorizing Model"
            dropdownOptions={categorizingModels}
            submitEndpoint="http://127.0.0.1:5001/api/classify-weakest-link"
            systemStatusEndpoint="/api/system-status"
          />
        )}
      </Box>
    </Box>
  );
}
