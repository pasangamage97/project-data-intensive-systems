import React from "react";
import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import Container from "@mui/material/Container";

export default function Navbar() {
  return (
    <AppBar position="static">
      <Container maxWidth="md">
        <Toolbar>
          <Typography variant="h5" component="div" sx={{ flexGrow: 1 }}>
            Project in Data Intensive Systems
          </Typography>
        </Toolbar>
      </Container>
    </AppBar>
  );
}
