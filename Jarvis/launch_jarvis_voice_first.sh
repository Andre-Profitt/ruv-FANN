#!/bin/bash

# JARVIS Ultimate Voice-First Launcher
# A sophisticated AI assistant with voice-first interaction and swarm intelligence

echo "🎙️  JARVIS Ultimate Voice-First Assistant"
echo "========================================="
echo

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Install/upgrade dependencies
echo -e "${BLUE}Checking dependencies...${NC}"

# Core dependencies
pip install --quiet --upgrade \
    speechrecognition \
    pyttsx3 \
    sounddevice \
    webrtcvad \
    numpy \
    pandas \
    scikit-learn \
    torch \
    asyncio \
    python-dotenv

# Additional voice dependencies
pip install --quiet --upgrade \
    pyaudio \
    google-cloud-speech \
    azure-cognitiveservices-speech \
    boto3  # For AWS Polly

# Swarm integration
echo -e "${BLUE}Checking ruv-swarm integration...${NC}"
if ! command -v npx &> /dev/null; then
    echo -e "${RED}Error: Node.js/npm not found. Please install Node.js for swarm features.${NC}"
    echo "Visit: https://nodejs.org/"
    exit 1
fi

# Initialize ruv-swarm if not already done
if [ ! -f ".swarm_initialized" ]; then
    echo -e "${YELLOW}Initializing ruv-swarm...${NC}"
    npx ruv-swarm init hierarchical 20
    touch .swarm_initialized
fi

# Check for API keys
echo -e "${BLUE}Checking API configuration...${NC}"
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cat > .env << EOF
# JARVIS Configuration
JARVIS_NAME="JARVIS"
JARVIS_PERSONALITY="professional_friendly"

# Speech Recognition (choose one or multiple)
GOOGLE_CLOUD_API_KEY=""
AZURE_SPEECH_KEY=""
AZURE_SPEECH_REGION=""
AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY=""
AWS_REGION=""

# AI Models (optional but recommended)
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""
GOOGLE_AI_API_KEY=""

# Voice Settings
VOICE_LANGUAGE="en-US"
VOICE_GENDER="male"
VOICE_SPEED=1.0

# Features
ENABLE_SWARM=true
ENABLE_PREDICTIONS=true
ENABLE_AUTOMATION=true
ENABLE_LEARNING=true

# Performance
MAX_RESPONSE_TIME_MS=200
CACHE_SIZE=1000
PREDICTION_HORIZON_HOURS=24
EOF
    echo -e "${YELLOW}Please edit .env file with your API keys${NC}"
fi

# Test microphone
echo -e "${BLUE}Testing audio devices...${NC}"
python3 -c "
import sounddevice as sd
print('Available audio devices:')
print(sd.query_devices())
print('\nDefault input device:', sd.query_devices(kind='input')['name'])
print('Default output device:', sd.query_devices(kind='output')['name'])
" 2>/dev/null || echo -e "${YELLOW}Warning: Audio device test failed${NC}"

# Create run configuration
cat > jarvis_config.json << EOF
{
    "name": "JARVIS",
    "version": "Ultimate Voice-First",
    "personality": {
        "type": "professional_friendly",
        "humor_level": 0.3,
        "formality": 0.6,
        "proactivity": 0.8
    },
    "features": {
        "voice_first": true,
        "anticipatory_ai": true,
        "swarm_intelligence": true,
        "continuous_learning": true,
        "proactive_automation": true
    },
    "performance": {
        "max_response_time": 200,
        "cache_size": 1000,
        "prediction_horizon": 24
    },
    "voice": {
        "wake_words": ["jarvis", "hey jarvis", "okay jarvis"],
        "continuous_listening": true,
        "interrupt_handling": true,
        "emotion_detection": true
    }
}
EOF

# Display startup message
clear
echo -e "${GREEN}"
echo "     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗"
echo "     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝"
echo "     ██║███████║██████╔╝██║   ██║██║███████╗"
echo "██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║"
echo "╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║"
echo " ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝"
echo -e "${NC}"
echo -e "${BLUE}Ultimate Voice-First Personal Assistant${NC}"
echo "========================================="
echo
echo -e "${GREEN}Features:${NC}"
echo "  • Voice-first interaction with continuous listening"
echo "  • Anticipatory AI that predicts your needs"
echo "  • Swarm intelligence for distributed processing"
echo "  • Continuous learning from interactions"
echo "  • Proactive automation and suggestions"
echo
echo -e "${YELLOW}Commands:${NC}"
echo "  • Say 'Hey JARVIS' to start a conversation"
echo "  • Say 'Stop' or 'Wait' to interrupt"
echo "  • Say 'Thank you' or 'Goodbye' to end"
echo "  • Press Ctrl+C to shutdown"
echo
echo -e "${BLUE}Starting JARVIS...${NC}"
echo

# Set environment variables
export JARVIS_CONFIG_PATH="jarvis_config.json"
export PYTHONUNBUFFERED=1

# Launch JARVIS
if [ "$1" == "--debug" ]; then
    echo -e "${YELLOW}Running in debug mode...${NC}"
    export JARVIS_DEBUG=1
    python3 jarvis_ultimate_voice_first.py
else
    python3 jarvis_ultimate_voice_first.py 2>/dev/null
fi

# Cleanup on exit
echo
echo -e "${BLUE}Shutting down JARVIS...${NC}"
deactivate 2>/dev/null
echo -e "${GREEN}Goodbye!${NC}"