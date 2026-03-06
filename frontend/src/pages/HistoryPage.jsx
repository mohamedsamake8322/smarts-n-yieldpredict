import {
    Delete as DeleteIcon,
    History as HistoryIcon,
    Visibility as ViewIcon,
} from '@mui/icons-material'
import {
    Alert,
    Box,
    Button,
    Card,
    CardContent,
    CardMedia,
    Chip,
    CircularProgress,
    Container,
    Grid,
    Paper,
    Tab,
    Tabs,
    Typography,
} from '@mui/material'
import { format } from 'date-fns'
import React, { useEffect, useState } from 'react'
import api from '../services/api'

const HistoryPage = () => {
  const [tabValue, setTabValue] = useState(0)
  const [detections, setDetections] = useState([])
  const [chats, setChats] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    loadHistory()
  }, [])

  const loadHistory = async () => {
    try {
      setLoading(true)
      const response = await api.get('/api/history/user_123') // TODO: Récupérer depuis l'auth
      setDetections(response.data.detections || [])
      setChats(response.data.chats || [])
    } catch (err) {
      setError(err.response?.data?.detail || 'Erreur lors du chargement de l\'historique')
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteDetection = async (detectionId) => {
    try {
      await api.delete(`/api/detection/${detectionId}?user_id=user_123`)
      setDetections(detections.filter((d) => d.id !== detectionId))
    } catch (err) {
      setError('Erreur lors de la suppression')
    }
  }

  const getSeverityColor = (severity) => {
    const colors = {
      low: 'success',
      moderate: 'warning',
      severe: 'error',
      critical: 'error',
    }
    return colors[severity] || 'default'
  }

  if (loading) {
    return (
      <Container maxWidth="lg">
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      </Container>
    )
  }

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          <HistoryIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Historique
        </Typography>
        <Typography variant="body1" color="text.secondary" align="center" sx={{ mb: 4 }}>
          Consultez vos détections et conversations précédentes
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        <Paper sx={{ mb: 3 }}>
          <Tabs value={tabValue} onChange={(e, newValue) => setTabValue(newValue)}>
            <Tab label={`Détections (${detections.length})`} />
            <Tab label={`Conversations (${chats.length})`} />
          </Tabs>
        </Paper>

        {tabValue === 0 && (
          <Grid container spacing={3}>
            {detections.length === 0 ? (
              <Grid item xs={12}>
                <Paper sx={{ p: 4, textAlign: 'center' }}>
                  <Typography variant="body1" color="text.secondary">
                    Aucune détection enregistrée
                  </Typography>
                </Paper>
              </Grid>
            ) : (
              detections.map((detection) => (
                <Grid item xs={12} md={6} key={detection.id}>
                  <Card>
                    {detection.image_path && (
                      <CardMedia
                        component="img"
                        height="200"
                        image={`/api/images/${detection.id}`} // TODO: Endpoint pour servir les images
                        alt={detection.plant_name}
                        sx={{ objectFit: 'cover' }}
                      />
                    )}
                    <CardContent>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                        <Typography variant="h6">{detection.plant_name}</Typography>
                        <Chip
                          label={detection.severity}
                          color={getSeverityColor(detection.severity)}
                          size="small"
                        />
                      </Box>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        {detection.plant_scientific_name}
                      </Typography>
                      {detection.diseases && detection.diseases.length > 0 && (
                        <Box sx={{ mt: 2 }}>
                          <Typography variant="body2" fontWeight="bold" gutterBottom>
                            Maladies :
                          </Typography>
                          {detection.diseases.map((disease, i) => (
                            <Chip
                              key={i}
                              label={disease.name}
                              size="small"
                              sx={{ mr: 0.5, mb: 0.5 }}
                            />
                          ))}
                        </Box>
                      )}
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 2 }}>
                        {format(new Date(detection.created_at), 'PPpp')}
                      </Typography>
                      <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                        <Button
                          size="small"
                          startIcon={<ViewIcon />}
                          onClick={() => {
                            // TODO: Ouvrir une modal avec les détails
                          }}
                        >
                          Voir
                        </Button>
                        <Button
                          size="small"
                          color="error"
                          startIcon={<DeleteIcon />}
                          onClick={() => handleDeleteDetection(detection.id)}
                        >
                          Supprimer
                        </Button>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))
            )}
          </Grid>
        )}

        {tabValue === 1 && (
          <Box>
            {chats.length === 0 ? (
              <Paper sx={{ p: 4, textAlign: 'center' }}>
                <Typography variant="body1" color="text.secondary">
                  Aucune conversation enregistrée
                </Typography>
              </Paper>
            ) : (
              chats.map((chat) => (
                <Paper key={chat.id} sx={{ p: 2, mb: 2 }}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    {format(new Date(chat.created_at), 'PPpp')}
                  </Typography>
                  <Typography variant="body1" fontWeight="bold" gutterBottom>
                    Vous : {chat.message}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Assistant : {chat.response}
                  </Typography>
                </Paper>
              ))
            )}
          </Box>
        )}
      </Box>
    </Container>
  )
}

export default HistoryPage

