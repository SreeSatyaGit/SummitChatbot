# SummitChatbot

An AI-powered conversational onboarding system designed for student-athletes, featuring multilingual support, fine-tuned language models, and intelligent data extraction.

## Overview

SummitChatbot is a comprehensive chatbot solution that guides users through an onboarding process, collecting personal, academic, athletic, and career information through natural conversation. The system combines multiple AI technologies including fine-tuned language models, RAG (Retrieval-Augmented Generation), and rule-based extraction to provide intelligent, context-aware responses.

## Architecture

The system consists of several interconnected services:

- **Onboarding API**: Main conversational interface with intelligent field extraction
- **RAG Server**: Fine-tuned language model service with knowledge base integration
- **Rasa Framework**: Intent recognition and dialogue management
- **Quality Monitor**: Response quality tracking and analytics
- **Multilingual Support**: Multi-language conversation capabilities

## Key Features

- **Conversational Onboarding**: Natural language data collection for student profiles
- **Intelligent Extraction**: Automatic parsing of user input into structured data
- **Multilingual Support**: Conversations in multiple languages
- **Fine-tuned Models**: Custom-trained language models for domain-specific responses
- **Quality Monitoring**: Real-time response quality assessment
- **Docker Deployment**: Containerized services with Docker Compose
- **Knowledge Base Integration**: RAG-powered responses using curated knowledge

## Technology Stack

- **Backend**: FastAPI, Python
- **AI/ML**: Hugging Face Transformers, RAG, Fine-tuning
- **NLP**: Rasa, Sentence Transformers
- **Deployment**: Docker, Docker Swarm
- **Monitoring**: Custom quality assessment tools

## Quick Start

1. **Prerequisites**: Docker, NVIDIA GPU (for model inference)
2. **Setup**: Configure API keys in `secrets/` directory
3. **Deploy**: Run `docker-compose up` to start all services
4. **Access**: API available at `http://localhost:8000`

## Project Structure

```
summit/
├── service/           # Main onboarding API
├── fine_tuning/      # Model training and RAG services
├── rasa/             # Rasa chatbot framework
├── scripts/          # Deployment and testing scripts
└── secrets/          # API keys and configuration
```

## Services

- **Port 8000**: Onboarding API (main interface)
- **Port 5055**: Rasa Action Server
- **Internal**: RAG Server (model inference)

## Use Cases

- Student-athlete onboarding and profile creation
- Academic and athletic information collection
- Career experience documentation
- Multilingual user support
- Automated data validation and structuring

## Development

The project supports both local development and production deployment with Docker Swarm, featuring automated CI/CD pipelines and secure secret management.
