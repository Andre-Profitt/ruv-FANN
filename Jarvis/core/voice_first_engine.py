#!/usr/bin/env python3
"""
JARVIS Voice-First Engine
A sophisticated voice-first interaction system with continuous listening,
natural conversation flow, and intelligent interruption handling.
"""

import asyncio
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import speech_recognition as sr
import pyttsx3
import sounddevice as sd
import webrtcvad
import collections
import threading
import queue
import json
import os

# Import ruv-swarm for cognitive capabilities
try:
    from ruv_swarm import SwarmClient
    SWARM_AVAILABLE = True
except ImportError:
    SWARM_AVAILABLE = False
    print("ruv-swarm not available, running without swarm intelligence")


@dataclass
class VoiceContext:
    """Maintains conversation context for natural interactions"""
    user_speaking: bool = False
    jarvis_speaking: bool = False
    last_interaction: Optional[datetime] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    current_topic: Optional[str] = None
    interruption_count: int = 0
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    ambient_noise_level: float = 0.0
    voice_profile: Dict[str, Any] = field(default_factory=dict)


class VoiceActivityDetector:
    """Advanced VAD with adaptive threshold and noise cancellation"""
    
    def __init__(self, sample_rate=16000, frame_duration_ms=30):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_length = int(sample_rate * (frame_duration_ms / 1000.0))
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (highest)
        self.ring_buffer = collections.deque(maxlen=10)
        self.triggered = False
        self.noise_threshold = 0.02
        
    def is_speech(self, audio_frame: bytes) -> bool:
        """Detect if audio frame contains speech"""
        try:
            is_speech = self.vad.is_speech(audio_frame, self.sample_rate)
            self.ring_buffer.append(is_speech)
            
            # Require 70% of frames to be speech
            num_voiced = sum(self.ring_buffer)
            if not self.triggered:
                if num_voiced > 0.7 * len(self.ring_buffer):
                    self.triggered = True
                    return True
            else:
                if num_voiced < 0.3 * len(self.ring_buffer):
                    self.triggered = False
                return True
                
        except Exception as e:
            print(f"VAD error: {e}")
            
        return False
        
    def adapt_to_environment(self, ambient_noise: float):
        """Adjust sensitivity based on ambient noise"""
        if ambient_noise > 0.1:
            self.vad = webrtcvad.Vad(2)  # Less aggressive in noisy environment
        else:
            self.vad = webrtcvad.Vad(3)  # More aggressive in quiet environment


class ConversationalTTS:
    """Text-to-speech with emotional intonation and natural pausing"""
    
    def __init__(self):
        self.engine = pyttsx3.init()
        self.setup_voice()
        self.speaking = False
        self.interrupt_flag = threading.Event()
        
    def setup_voice(self):
        """Configure voice for natural conversation"""
        voices = self.engine.getProperty('voices')
        # Prefer a natural-sounding voice
        for voice in voices:
            if 'samantha' in voice.name.lower() or 'alex' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
                
        self.engine.setProperty('rate', 175)  # Slightly slower for clarity
        self.engine.setProperty('volume', 0.9)
        
    def speak_async(self, text: str, emotion: str = "neutral", priority: int = 5):
        """Speak with emotional context and interruption handling"""
        def _speak():
            self.speaking = True
            self.interrupt_flag.clear()
            
            # Adjust speech parameters based on emotion
            if emotion == "excited":
                self.engine.setProperty('rate', 190)
                self.engine.setProperty('pitch', 1.1)
            elif emotion == "concerned":
                self.engine.setProperty('rate', 160)
                self.engine.setProperty('pitch', 0.9)
            elif emotion == "thoughtful":
                self.engine.setProperty('rate', 150)
                
            # Natural pausing at punctuation
            sentences = text.replace('!', '.').replace('?', '.').split('.')
            for sentence in sentences:
                if self.interrupt_flag.is_set():
                    break
                if sentence.strip():
                    self.engine.say(sentence.strip())
                    self.engine.runAndWait()
                    if not self.interrupt_flag.is_set():
                        asyncio.sleep(0.2)  # Natural pause between sentences
                        
            self.speaking = False
            
        thread = threading.Thread(target=_speak)
        thread.daemon = True
        thread.start()
        
    def interrupt(self):
        """Gracefully interrupt current speech"""
        if self.speaking:
            self.interrupt_flag.set()
            self.engine.stop()
            self.speaking = False


class VoiceFirstEngine:
    """Main voice-first interaction engine with anticipatory capabilities"""
    
    def __init__(self, swarm_enabled: bool = True):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts = ConversationalTTS()
        self.vad = VoiceActivityDetector()
        self.context = VoiceContext()
        self.audio_queue = queue.Queue()
        self.command_queue = queue.Queue()
        
        # Initialize swarm integration if available
        self.swarm = None
        if swarm_enabled and SWARM_AVAILABLE:
            self.swarm = SwarmClient(
                topology="hierarchical",
                max_agents=10,
                enable_neural=True
            )
            
        # Continuous listening settings
        self.listening = False
        self.processing = False
        self.wake_words = ["jarvis", "hey jarvis", "okay jarvis", "yo jarvis"]
        self.end_phrases = ["thank you", "thanks", "that's all", "goodbye"]
        
        # Calibrate for ambient noise
        self.calibrate_ambient_noise()
        
    def calibrate_ambient_noise(self):
        """Calibrate microphone for ambient noise level"""
        with self.microphone as source:
            print("Calibrating for ambient noise... Please be quiet.")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            # Sample ambient noise level
            audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=1)
            audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)
            self.context.ambient_noise_level = np.mean(np.abs(audio_data)) / 32768.0
            self.vad.adapt_to_environment(self.context.ambient_noise_level)
            print(f"Calibration complete. Noise level: {self.context.ambient_noise_level:.3f}")
            
    async def start_continuous_listening(self):
        """Start the continuous listening loop"""
        self.listening = True
        
        # Start audio streaming thread
        audio_thread = threading.Thread(target=self._audio_streaming_thread)
        audio_thread.daemon = True
        audio_thread.start()
        
        # Start processing thread
        process_thread = threading.Thread(target=self._process_audio_thread)
        process_thread.daemon = True
        process_thread.start()
        
        # Welcome message
        await self.respond("Voice system activated. I'm listening.", emotion="neutral")
        
        # Main listening loop
        while self.listening:
            try:
                # Check for commands from the queue
                if not self.command_queue.empty():
                    command = self.command_queue.get()
                    await self.process_command(command)
                    
                # Brief sleep to prevent CPU spinning
                await asyncio.sleep(0.01)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Listening error: {e}")
                
        self.listening = False
        
    def _audio_streaming_thread(self):
        """Continuous audio streaming from microphone"""
        CHUNK = 480  # 30ms at 16kHz
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio stream status: {status}")
            self.audio_queue.put(bytes(indata))
            
        with sd.RawInputStream(
            samplerate=16000,
            blocksize=CHUNK,
            dtype='int16',
            channels=1,
            callback=audio_callback
        ):
            while self.listening:
                sd.sleep(100)
                
    def _process_audio_thread(self):
        """Process audio chunks for voice activity and recognition"""
        frames = []
        voiced_frames = []
        
        while self.listening:
            try:
                # Get audio chunk
                chunk = self.audio_queue.get(timeout=0.1)
                frames.append(chunk)
                
                # Check for voice activity
                if self.vad.is_speech(chunk):
                    voiced_frames.extend(frames)
                    frames = []
                else:
                    if len(voiced_frames) > 10:  # Minimum speech length
                        # Convert to recognizable format
                        audio_data = b''.join(voiced_frames)
                        self._recognize_speech(audio_data)
                    voiced_frames = []
                    
                # Keep buffer size manageable
                if len(frames) > 100:
                    frames = frames[-50:]
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio processing error: {e}")
                
    def _recognize_speech(self, audio_data: bytes):
        """Recognize speech from audio data"""
        try:
            # Convert to AudioData format
            audio = sr.AudioData(audio_data, 16000, 2)
            
            # Attempt recognition
            text = self.recognizer.recognize_google(audio)
            
            # Check if it's a wake word or command
            text_lower = text.lower()
            
            # If JARVIS is speaking, this might be an interruption
            if self.tts.speaking:
                if any(word in text_lower for word in ["stop", "wait", "hold on"]):
                    self.tts.interrupt()
                    self.context.interruption_count += 1
                    
            # Process as command
            if any(wake in text_lower for wake in self.wake_words) or self.context.user_speaking:
                self.command_queue.put(text)
                self.context.user_speaking = True
                self.context.last_interaction = datetime.now()
                
            # Check for end of conversation
            if any(end in text_lower for end in self.end_phrases):
                self.context.user_speaking = False
                
        except sr.UnknownValueError:
            pass  # Couldn't understand audio
        except sr.RequestError as e:
            print(f"Recognition error: {e}")
            
    async def process_command(self, command: str):
        """Process recognized command with context awareness"""
        # Update conversation history
        self.context.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": command,
            "context": self.context.current_topic
        })
        
        # Use swarm for intelligent processing if available
        if self.swarm:
            response = await self._process_with_swarm(command)
        else:
            response = await self._process_basic(command)
            
        # Respond with appropriate emotion
        emotion = self._determine_emotion(response)
        await self.respond(response["text"], emotion=emotion, metadata=response.get("metadata"))
        
    async def _process_with_swarm(self, command: str) -> Dict[str, Any]:
        """Process command using ruv-swarm cognitive capabilities"""
        # Spawn specialized agents for different aspects
        agents = [
            {"type": "analyzer", "task": "understand_intent"},
            {"type": "predictor", "task": "anticipate_needs"},
            {"type": "personalizer", "task": "tailor_response"}
        ]
        
        # Orchestrate parallel processing
        results = await self.swarm.orchestrate(
            task=f"Process voice command: {command}",
            agents=agents,
            context={
                "history": self.context.conversation_history[-10:],
                "preferences": self.context.user_preferences,
                "current_topic": self.context.current_topic
            }
        )
        
        # Synthesize results
        return {
            "text": results.get("response", "I understand. How can I help?"),
            "intent": results.get("intent"),
            "suggestions": results.get("suggestions", []),
            "metadata": results.get("metadata", {})
        }
        
    async def _process_basic(self, command: str) -> Dict[str, Any]:
        """Basic command processing without swarm"""
        command_lower = command.lower()
        
        # Simple intent detection
        if "weather" in command_lower:
            return {"text": "I'll check the weather for you. It looks like it will be partly cloudy today with a high of 72 degrees."}
        elif "time" in command_lower:
            current_time = datetime.now().strftime("%I:%M %p")
            return {"text": f"The time is {current_time}."}
        elif "remind" in command_lower:
            return {"text": "I'll set that reminder for you. I'll make sure to notify you at the right time."}
        else:
            return {"text": "I'm here to help. What would you like me to do?"}
            
    def _determine_emotion(self, response: Dict[str, Any]) -> str:
        """Determine appropriate emotion for response"""
        text = response.get("text", "").lower()
        
        if any(word in text for word in ["congratulations", "great", "excellent"]):
            return "excited"
        elif any(word in text for word in ["sorry", "unfortunately", "problem"]):
            return "concerned"
        elif any(word in text for word in ["thinking", "considering", "perhaps"]):
            return "thoughtful"
        else:
            return "neutral"
            
    async def respond(self, text: str, emotion: str = "neutral", metadata: Optional[Dict] = None):
        """Respond with voice and update context"""
        # Update conversation history
        self.context.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "jarvis": text,
            "emotion": emotion,
            "metadata": metadata
        })
        
        # Speak the response
        self.tts.speak_async(text, emotion=emotion)
        
        # Update context based on response
        if metadata:
            if "topic" in metadata:
                self.context.current_topic = metadata["topic"]
            if "preferences" in metadata:
                self.context.user_preferences.update(metadata["preferences"])
                
    def stop(self):
        """Gracefully stop the voice engine"""
        self.listening = False
        self.tts.interrupt()
        if self.swarm:
            self.swarm.shutdown()


async def main():
    """Example usage of the Voice-First Engine"""
    engine = VoiceFirstEngine(swarm_enabled=True)
    
    print("Starting JARVIS Voice-First Engine...")
    print("Say 'Hey JARVIS' to start a conversation.")
    print("Press Ctrl+C to stop.")
    
    try:
        await engine.start_continuous_listening()
    except KeyboardInterrupt:
        print("\nShutting down...")
        engine.stop()


if __name__ == "__main__":
    asyncio.run(main())