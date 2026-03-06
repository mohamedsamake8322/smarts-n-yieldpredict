import React, { useState, useRef, useEffect } from 'react'
import {
  Container,
  Typography,
  Box,
  TextField,
  Button,
  Paper,
  CircularProgress,
  Alert,
  Chip,
  Avatar,
} from '@mui/material'
import {
  Send as SendIcon,
  SmartToy as BotIcon,
  Person as PersonIcon,
} from '@mui/icons-material'
import api from '../services/api'

const ChatPage = () => {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async () => {
    if (!input.trim() || loading) return

    const userMessage = input.trim()
    setInput('')
    setError(null)

    // Ajouter le message de l'utilisateur
    const newUserMessage = {
      role: 'user',
      content: userMessage,
      timestamp: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, newUserMessage])
    setLoading(true)

    try {
      const response = await api.post('/api/chat', {
        message: userMessage,
        user_id: 'user_123', // TODO: Récupérer depuis l'auth
      })

      const botMessage = {
        role: 'assistant',
        content: response.data.response,
        suggestions: response.data.suggestions,
        timestamp: response.data.timestamp,
      }
      setMessages((prev) => [...prev, botMessage])
    } catch (err) {
      setError(err.response?.data?.detail || 'Erreur lors de l\'envoi du message')
    } finally {
      setLoading(false)
    }
  }

  const handleSuggestionClick = (suggestion) => {
    setInput(suggestion)
  }

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Assistant Agricole
        </Typography>
        <Typography variant="body1" color="text.secondary" align="center" sx={{ mb: 4 }}>
          Posez vos questions sur les plantes, maladies, traitements et pratiques agricoles
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        <Paper
          sx={{
            height: '60vh',
            display: 'flex',
            flexDirection: 'column',
            mb: 2,
            overflow: 'hidden',
          }}
        >
          <Box
            sx={{
              flexGrow: 1,
              overflowY: 'auto',
              p: 2,
              bgcolor: 'background.default',
            }}
          >
            {messages.length === 0 && (
              <Box sx={{ textAlign: 'center', py: 4, color: 'text.secondary' }}>
                <BotIcon sx={{ fontSize: 64, mb: 2, opacity: 0.5 }} />
                <Typography variant="body1">
                  Bonjour ! Je suis votre assistant agricole. Comment puis-je vous aider ?
                </Typography>
                <Typography variant="body2" sx={{ mt: 2 }}>
                  Exemples de questions :
                </Typography>
                <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 1, justifyContent: 'center' }}>
                  {['Comment traiter le mildiou ?', 'Quand arroser ?', 'Quel engrais utiliser ?'].map(
                    (q) => (
                      <Chip
                        key={q}
                        label={q}
                        onClick={() => handleSuggestionClick(q)}
                        sx={{ cursor: 'pointer' }}
                      />
                    )
                  )}
                </Box>
              </Box>
            )}

            {messages.map((message, index) => (
              <Box
                key={index}
                sx={{
                  display: 'flex',
                  justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start',
                  mb: 2,
                }}
              >
                <Box
                  sx={{
                    display: 'flex',
                    flexDirection: message.role === 'user' ? 'row-reverse' : 'row',
                    alignItems: 'flex-start',
                    maxWidth: '70%',
                    gap: 1,
                  }}
                >
                  <Avatar
                    sx={{
                      bgcolor: message.role === 'user' ? 'primary.main' : 'secondary.main',
                    }}
                  >
                    {message.role === 'user' ? <PersonIcon /> : <BotIcon />}
                  </Avatar>
                  <Paper
                    sx={{
                      p: 2,
                      bgcolor: message.role === 'user' ? 'primary.main' : 'background.paper',
                      color: message.role === 'user' ? 'white' : 'text.primary',
                    }}
                  >
                    <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                      {message.content}
                    </Typography>
                    {message.suggestions && message.suggestions.length > 0 && (
                      <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                        {message.suggestions.map((suggestion, i) => (
                          <Chip
                            key={i}
                            label={suggestion}
                            onClick={() => handleSuggestionClick(suggestion)}
                            size="small"
                            sx={{
                              cursor: 'pointer',
                              bgcolor: message.role === 'user' ? 'rgba(255,255,255,0.2)' : 'primary.light',
                              color: message.role === 'user' ? 'white' : 'white',
                              '&:hover': {
                                bgcolor: message.role === 'user' ? 'rgba(255,255,255,0.3)' : 'primary.main',
                              },
                            }}
                          />
                        ))}
                      </Box>
                    )}
                  </Paper>
                </Box>
              </Box>
            ))}

            {loading && (
              <Box sx={{ display: 'flex', justifyContent: 'flex-start', mb: 2 }}>
                <Avatar sx={{ bgcolor: 'secondary.main' }}>
                  <BotIcon />
                </Avatar>
                <Paper sx={{ p: 2, ml: 1 }}>
                  <CircularProgress size={20} />
                </Paper>
              </Box>
            )}

            <div ref={messagesEndRef} />
          </Box>

          <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <TextField
                fullWidth
                placeholder="Posez votre question..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    handleSend()
                  }
                }}
                disabled={loading}
                multiline
                maxRows={3}
              />
              <Button
                variant="contained"
                onClick={handleSend}
                disabled={loading || !input.trim()}
                startIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
                sx={{ minWidth: 100 }}
              >
                Envoyer
              </Button>
            </Box>
          </Box>
        </Paper>
      </Box>
    </Container>
  )
}

export default ChatPage





