import React, { useState, useEffect } from "react";
import { Tabs, Tab, Box } from "@mui/material";
import FormComponent from "./FormComponent";

export default function TabView() {
  const [tabValue, setTabValue] = useState(0);
  const [models, setModels] = useState([]);
  const [categorizingModels, setCategorizingModels] = useState([]);

  useEffect(() => {
    fetch("http://127.0.0.1:5001/api/get-models")
      .then((res) => res.json())
      .then((data) => {
        setModels(data.models.map((name) => ({ label: name, value: name })));
        setCategorizingModels(data.categorizingModels.map((name) => ({ label: name, value: name })));
      })
      .catch((err) => console.error("Error fetching models:", err));
  }, []);

  // Fetch dropdown data when component mounts
//   useEffect(() => {
//     const fetchModels = async () => {
//       try {
//         const response = await fetch("http://127.0.0.1:5001/api/models");
//         if (response.ok) {
//           const data = await response.json();
//           setModels(data);
//         }
//       } catch (error) {
//         console.error("Error fetching models:", error);
//       }
//     };

//     const fetchCategorizingModels = async () => {
//       try {
//         const response = await fetch("http://127.0.0.1:5001/api/categorizingModels");
//         if (response.ok) {
//           const data = await response.json();
//           setCategorizingModels(data);
//         }
//       } catch (error) {
//         console.error("Error fetching categorizing models:", error);
//       }
//     };

//     fetchModels();
//     fetchCategorizingModels();
//   }, []);

  return (
    <Box sx={{ width: "100%" }}>
      <Tabs value={tabValue} onChange={(event, newValue) => setTabValue(newValue)}>
        <Tab label="Regression" />
        <Tab label="Classifier" />
      </Tabs>

      <Box sx={{ p: 2 }}>
        {tabValue === 0 && (
          <FormComponent
            title="Select a Model"
            dropdownOptions={models}
            submitEndpoint="predict"
            systemStatusEndpoint="/api/system-status"
          />
        )}
        {tabValue === 1 && (
          <FormComponent
            title="Select a Classifier Model"
            dropdownOptions={categorizingModels}
            submitEndpoint="classify-weakest-link"
            systemStatusEndpoint="/api/system-status"
          />
        )}
      </Box>
    </Box>
  );
}
