import React, { useState } from "react";
import { Tabs, Tab, Box, Typography, Container } from "@mui/material";

export default function TabView() {
  const [activeTab, setActiveTab] = useState(0);

  const handleChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  return (
    <Container maxWidth="md">
      {/* Tabs Navigation */}
      <Tabs value={activeTab} onChange={handleChange}>
        <Tab label="Regression" />
        <Tab label="Classification" />
      </Tabs>

      {/* Tab Content */}
      <Box sx={{ mt: 2, p: 2 }}>
        {activeTab === 0 && (
          <Typography variant="h6">This is the content for Tab One.</Typography>
        )}
        {activeTab === 1 && (
          <Typography variant="h6">This is the content for Tab Two.</Typography>
        )}
      </Box>
    </Container>
  );
}
