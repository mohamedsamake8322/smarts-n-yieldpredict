# Overview

This is a multilingual voice assistant web application built with Flask that processes documents and provides AI-powered conversational capabilities. The system allows users to upload various document formats (PDF, Excel, CSV, JSON, TXT), automatically detects the language of the content, stores documents in a vector-based search system, and provides voice interaction through speech recognition and synthesis. The application uses OpenAI's GPT API for intelligent responses and can operate in French, English, and Spanish.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Single-page application** using vanilla JavaScript with Bootstrap for UI components
- **Responsive design** with dark theme support via Bootstrap's data-bs-theme
- **Voice interaction capabilities** using Web Speech API for both recognition and synthesis
- **Real-time chat interface** with animated message bubbles and typing indicators
- **File upload interface** with drag-and-drop support and progress tracking
- **Document search functionality** integrated into the main interface

## Backend Architecture
- **Flask web framework** with session management and security middleware (ProxyFix)
- **Modular service architecture** with separate classes for different responsibilities:
  - `DocumentProcessor`: Handles multiple file format processing (PDF, Excel, CSV, JSON, TXT)
  - `VoiceAssistant`: Manages OpenAI API integration for AI responses
  - `VectorStore`: In-memory vector database for document storage and retrieval using TF-IDF
  - `LanguageDetector`: Rule-based language detection for French, English, and Spanish
- **RESTful API design** with endpoints for file upload, chat, search, and document management
- **File handling** using temporary storage with security validation and size limits (16MB)

## Data Storage Solutions
- **In-memory vector store** using TF-IDF (Term Frequency-Inverse Document Frequency) for document indexing
- **Session-based state management** for user preferences and chat history
- **Temporary file storage** in system temp directory for uploaded documents
- **No persistent database** - all data is stored in memory during runtime

## Authentication and Authorization
- **Basic session management** using Flask sessions with configurable secret key
- **File upload validation** with allowed extensions and size restrictions
- **No user authentication system** - single-user application model

# External Dependencies

## AI Services
- **OpenAI GPT API** for conversational AI responses with multilingual support
- Configurable via `OPENAI_API_KEY` environment variable
- Supports context-aware responses using document content

## Document Processing Libraries
- **PyPDF2** for PDF text extraction
- **pandas** for Excel file processing (xlsx, xls)
- **csv module** for CSV file handling
- Built-in JSON and text file processing

## Frontend Libraries
- **Bootstrap 5** with Replit dark theme for UI components
- **Font Awesome 6** for icons and visual elements
- **Web Speech API** for voice recognition and text-to-speech (browser-native)

## Python Dependencies
- **Flask** web framework with Werkzeug utilities
- **numpy** for vector operations in the search system
- **Standard library modules** for file handling, logging, and text processing

## Configuration
- **Environment-based configuration** for API keys and session secrets
- **Development-friendly defaults** with debug mode enabled
- **Configurable upload limits** and allowed file types