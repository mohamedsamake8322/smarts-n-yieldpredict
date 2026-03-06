import React from 'react'
import { Routes, Route } from 'react-router-dom'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'
import HomePage from './pages/HomePage'
import DetectionPage from './pages/DetectionPage'
import ChatPage from './pages/ChatPage'
import HistoryPage from './pages/HistoryPage'
import Layout from './components/Layout'

const theme = createTheme({
  palette: {
    primary: {
      main: '#2e7d32', // Vert agricole
      light: '#4caf50',
      dark: '#1b5e20',
    },
    secondary: {
      main: '#ff6f00',
      light: '#ff8f00',
      dark: '#e65100',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: [
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 12,
  },
})

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/detect" element={<DetectionPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/history" element={<HistoryPage />} />
        </Routes>
      </Layout>
    </ThemeProvider>
  )
}

export default App





