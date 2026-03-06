import React from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Container,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Paper,
} from '@mui/material'
import {
  CameraAlt as CameraIcon,
  Chat as ChatIcon,
  History as HistoryIcon,
  Science as ScienceIcon,
} from '@mui/icons-material'

const HomePage = () => {
  const navigate = useNavigate()

  const features = [
    {
      title: 'Détection Intelligente',
      description: 'Identifiez les plantes et diagnostiquez les maladies grâce à l\'IA',
      icon: <CameraIcon sx={{ fontSize: 48, color: 'primary.main' }} />,
      action: 'Démarrer la détection',
      path: '/detect',
    },
    {
      title: 'Assistant Conversationnel',
      description: 'Posez vos questions et obtenez des conseils personnalisés',
      icon: <ChatIcon sx={{ fontSize: 48, color: 'secondary.main' }} />,
      action: 'Parler à l\'assistant',
      path: '/chat',
    },
    {
      title: 'Historique',
      description: 'Consultez vos détections précédentes et suivez l\'évolution',
      icon: <HistoryIcon sx={{ fontSize: 48, color: 'primary.main' }} />,
      action: 'Voir l\'historique',
      path: '/history',
    },
  ]

  return (
    <Container maxWidth="lg">
      <Box sx={{ textAlign: 'center', my: 6 }}>
        <Typography
          variant="h2"
          component="h1"
          gutterBottom
          sx={{ fontWeight: 700, color: 'primary.main' }}
        >
          🌱 Agro-Scan
        </Typography>
        <Typography
          variant="h5"
          component="h2"
          color="text.secondary"
          gutterBottom
          sx={{ mb: 4 }}
        >
          Application intelligente de détection des plantes et maladies agricoles
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 800, mx: 'auto', mb: 4 }}>
          Une solution performante et accessible pour diagnostiquer vos cultures et obtenir
          des recommandations pratiques adaptées aux producteurs agricoles.
        </Typography>
      </Box>

      <Grid container spacing={4} sx={{ mb: 6 }}>
        {features.map((feature, index) => (
          <Grid item xs={12} md={4} key={index}>
            <Card
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                transition: 'transform 0.2s, box-shadow 0.2s',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: 6,
                },
              }}
            >
              <CardContent sx={{ flexGrow: 1, textAlign: 'center', pt: 4 }}>
                <Box sx={{ mb: 2 }}>{feature.icon}</Box>
                <Typography variant="h5" component="h3" gutterBottom>
                  {feature.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {feature.description}
                </Typography>
              </CardContent>
              <CardActions sx={{ justifyContent: 'center', pb: 3 }}>
                <Button
                  variant="contained"
                  onClick={() => navigate(feature.path)}
                  size="large"
                >
                  {feature.action}
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Paper sx={{ p: 4, bgcolor: 'primary.light', color: 'white', textAlign: 'center' }}>
        <ScienceIcon sx={{ fontSize: 64, mb: 2, opacity: 0.9 }} />
        <Typography variant="h5" gutterBottom>
          Technologie de pointe
        </Typography>
        <Typography variant="body1" sx={{ maxWidth: 600, mx: 'auto' }}>
          Notre application utilise des modèles de vision par ordinateur avancés et un assistant
          conversationnel intelligent pour vous offrir des diagnostics précis et des recommandations
          adaptées à vos cultures.
        </Typography>
      </Paper>
    </Container>
  )
}

export default HomePage





