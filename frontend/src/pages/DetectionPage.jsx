import React, { useState, useRef } from 'react'
import {
  Container,
  Typography,
  Box,
  Button,
  Paper,
  CircularProgress,
  Alert,
  Grid,
  Card,
  CardContent,
  Chip,
  Stepper,
  Step,
  StepLabel,
} from '@mui/material'
import {
  CameraAlt as CameraIcon,
  PhotoLibrary as PhotoLibraryIcon,
  CheckCircle as CheckCircleIcon,
} from '@mui/icons-material'
import Webcam from 'react-webcam'
import { useDropzone } from 'react-dropzone'
import api from '../services/api'

const DetectionPage = () => {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [activeStep, setActiveStep] = useState(0)
  const [imagePreview, setImagePreview] = useState(null)
  const webcamRef = useRef(null)

  const steps = ['Prendre/Téléverser une photo', 'Analyse en cours', 'Résultats']

  const handleFileUpload = async (file) => {
    setLoading(true)
    setError(null)
    setActiveStep(1)
    setImagePreview(URL.createObjectURL(file))

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('user_id', 'user_123') // TODO: Récupérer depuis l'auth

      const response = await api.post('/api/detect', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      setResult(response.data)
      setActiveStep(2)
    } catch (err) {
      setError(err.response?.data?.detail || 'Erreur lors de la détection')
      setActiveStep(0)
    } finally {
      setLoading(false)
    }
  }

  const capturePhoto = async () => {
    const imageSrc = webcamRef.current?.getScreenshot()
    if (imageSrc) {
      // Convertir base64 en blob
      const response = await fetch(imageSrc)
      const blob = await response.blob()
      const file = new File([blob], 'photo.jpg', { type: 'image/jpeg' })
      await handleFileUpload(file)
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp'],
    },
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        handleFileUpload(acceptedFiles[0])
      }
    },
    multiple: false,
  })

  const getSeverityColor = (severity) => {
    const colors = {
      low: 'success',
      moderate: 'warning',
      severe: 'error',
      critical: 'error',
    }
    return colors[severity] || 'default'
  }

  const resetDetection = () => {
    setResult(null)
    setError(null)
    setImagePreview(null)
    setActiveStep(0)
  }

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Détection Intelligente
        </Typography>
        <Typography variant="body1" color="text.secondary" align="center" sx={{ mb: 4 }}>
          Prenez une photo ou téléversez une image pour identifier la plante et diagnostiquer les maladies
        </Typography>

        <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {!result && (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Paper
                {...getRootProps()}
                sx={{
                  p: 4,
                  textAlign: 'center',
                  cursor: 'pointer',
                  border: '2px dashed',
                  borderColor: isDragActive ? 'primary.main' : 'grey.300',
                  bgcolor: isDragActive ? 'action.hover' : 'background.paper',
                  transition: 'all 0.2s',
                  '&:hover': {
                    borderColor: 'primary.main',
                    bgcolor: 'action.hover',
                  },
                }}
              >
                <input {...getInputProps()} />
                <PhotoLibraryIcon sx={{ fontSize: 64, color: 'primary.main', mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  {isDragActive ? 'Déposez l\'image ici' : 'Téléverser une image'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Glissez-déposez une image ou cliquez pour sélectionner
                </Typography>
              </Paper>
            </Grid>

            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h6" gutterBottom>
                  Prendre une photo
                </Typography>
                <Box sx={{ position: 'relative', mb: 2 }}>
                  <Webcam
                    audio={false}
                    ref={webcamRef}
                    screenshotFormat="image/jpeg"
                    videoConstraints={{
                      facingMode: 'environment',
                    }}
                    style={{
                      width: '100%',
                      maxWidth: '100%',
                      borderRadius: 8,
                    }}
                  />
                </Box>
                <Button
                  variant="contained"
                  startIcon={<CameraIcon />}
                  onClick={capturePhoto}
                  disabled={loading}
                  size="large"
                >
                  Capturer
                </Button>
              </Paper>
            </Grid>
          </Grid>
        )}

        {loading && (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', my: 4 }}>
            <CircularProgress size={60} sx={{ mb: 2 }} />
            <Typography variant="body1" color="text.secondary">
              Analyse de l'image en cours...
            </Typography>
          </Box>
        )}

        {result && (
          <Box>
            <Button
              variant="outlined"
              onClick={resetDetection}
              sx={{ mb: 3 }}
            >
              Nouvelle détection
            </Button>

            {imagePreview && (
              <Box sx={{ mb: 3, textAlign: 'center' }}>
                <img
                  src={imagePreview}
                  alt="Analyse"
                  style={{
                    maxWidth: '100%',
                    maxHeight: '400px',
                    borderRadius: 8,
                    boxShadow: 2,
                  }}
                />
              </Box>
            )}

            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Plante détectée
                    </Typography>
                    <Typography variant="h5" color="primary" gutterBottom>
                      {result.plant_info.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      {result.plant_info.scientific_name}
                    </Typography>
                    <Chip
                      label={`Confiance: ${(result.confidence_score * 100).toFixed(1)}%`}
                      color="primary"
                      size="small"
                    />
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Niveau de gravité
                    </Typography>
                    <Chip
                      label={result.overall_severity}
                      color={getSeverityColor(result.overall_severity)}
                      size="large"
                      sx={{ fontSize: '1rem', py: 2 }}
                    />
                  </CardContent>
                </Card>
              </Grid>

              {result.diseases && result.diseases.length > 0 && (
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Maladies détectées
                      </Typography>
                      {result.diseases.map((disease, index) => (
                        <Box key={index} sx={{ mb: 2 }}>
                          <Typography variant="subtitle1" fontWeight="bold">
                            {disease.name}
                          </Typography>
                          <Typography variant="body2" color="text.secondary" gutterBottom>
                            {disease.description}
                          </Typography>
                          <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                            <Chip
                              label={`Confiance: ${(disease.confidence * 100).toFixed(1)}%`}
                              size="small"
                            />
                            <Chip
                              label={disease.severity}
                              color={getSeverityColor(disease.severity)}
                              size="small"
                            />
                          </Box>
                        </Box>
                      ))}
                    </CardContent>
                  </Card>
                </Grid>
              )}

              {result.recommendations && result.recommendations.length > 0 && (
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        <CheckCircleIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                        Recommandations
                      </Typography>
                      {result.recommendations.map((rec, index) => (
                        <Box key={index} sx={{ mb: 3, p: 2, bgcolor: 'background.default', borderRadius: 2 }}>
                          <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                            {rec.title}
                          </Typography>
                          <Typography variant="body2" color="text.secondary" gutterBottom>
                            {rec.description}
                          </Typography>
                          {rec.steps && rec.steps.length > 0 && (
                            <Box sx={{ mt: 2 }}>
                              <Typography variant="body2" fontWeight="bold" gutterBottom>
                                Étapes :
                              </Typography>
                              <ul style={{ marginLeft: 20 }}>
                                {rec.steps.map((step, i) => (
                                  <li key={i}>
                                    <Typography variant="body2">{step}</Typography>
                                  </li>
                                ))}
                              </ul>
                            </Box>
                          )}
                          {rec.products && rec.products.length > 0 && (
                            <Box sx={{ mt: 2 }}>
                              <Typography variant="body2" fontWeight="bold">
                                Produits recommandés :
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                {rec.products.join(', ')}
                              </Typography>
                            </Box>
                          )}
                          {rec.organic_alternatives && rec.organic_alternatives.length > 0 && (
                            <Box sx={{ mt: 2 }}>
                              <Typography variant="body2" fontWeight="bold">
                                Alternatives biologiques :
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                {rec.organic_alternatives.join(', ')}
                              </Typography>
                            </Box>
                          )}
                        </Box>
                      ))}
                    </CardContent>
                  </Card>
                </Grid>
              )}
            </Grid>
          </Box>
        )}
      </Box>
    </Container>
  )
}

export default DetectionPage





