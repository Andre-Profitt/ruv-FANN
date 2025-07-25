#!/usr/bin/env python3
"""
JARVIS Ultimate Demo
Demonstrates the key capabilities of the voice-first personal assistant
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any

# Add core modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

# For demo purposes, we'll simulate some components
class DemoJARVIS:
    """Simplified JARVIS for demonstration"""
    
    def __init__(self):
        self.name = "JARVIS"
        self.start_time = datetime.now()
        print(f"🤖 {self.name} Ultimate Voice-First Assistant - Demo Mode")
        print("=" * 50)
        
    async def demo_voice_interaction(self):
        """Demonstrate voice-first interaction"""
        print("\n🎙️  Voice-First Interaction Demo")
        print("-" * 30)
        
        # Simulate continuous listening
        print("JARVIS: *listening continuously*")
        await asyncio.sleep(1)
        
        print('User: "Hey JARVIS"')
        await asyncio.sleep(0.5)
        
        print('JARVIS: "Good evening! I\'m here. How can I help you?"')
        await asyncio.sleep(1)
        
        print('User: "What\'s my schedule like tomorrow?"')
        await asyncio.sleep(0.5)
        
        print('JARVIS: "Tomorrow you have 4 meetings. Your day starts with the team standup at 9 AM."')
        print('        "I noticed you usually grab coffee before that - should I place your usual order for 8:45 AM pickup?"')
        await asyncio.sleep(2)
        
        print('User: "Yes, that would be great"')
        await asyncio.sleep(0.5)
        
        print('JARVIS: "Done! Your large cappuccino will be ready at the usual place."')
        print('        "Also, tomorrow\'s weather will be cooler - you might want to bring a jacket."')
        
    async def demo_anticipatory_ai(self):
        """Demonstrate anticipatory capabilities"""
        print("\n\n🧠 Anticipatory AI Demo")
        print("-" * 30)
        
        print("Scenario: It's 5:30 PM on a weekday")
        await asyncio.sleep(1)
        
        print('\nJARVIS: "I noticed you usually leave for your gym class around this time."')
        print('        "Traffic is lighter than usual today - you could leave in 10 minutes"')
        print('        "and still arrive early. Should I prepare your workout playlist?"')
        await asyncio.sleep(2)
        
        print("\nPredicted needs based on patterns:")
        predictions = [
            "🏃 Gym reminder (confidence: 92%)",
            "🎵 Workout playlist preparation (confidence: 87%)",
            "💧 Water bottle reminder (confidence: 78%)",
            "🚗 Traffic route optimization (confidence: 85%)"
        ]
        
        for pred in predictions:
            print(f"  • {pred}")
            await asyncio.sleep(0.3)
            
    async def demo_swarm_intelligence(self):
        """Demonstrate swarm processing"""
        print("\n\n🐝 Swarm Intelligence Demo")
        print("-" * 30)
        
        print('User: "Help me plan a dinner party for 8 people this Saturday"')
        await asyncio.sleep(1)
        
        print("\nSwarm agents collaborating:")
        agents = [
            ("Coordinator", "Orchestrating party planning task"),
            ("Researcher", "Finding recipes for dietary restrictions"),
            ("Analyzer", "Calculating quantities and timing"),
            ("Planner", "Creating shopping list and schedule"),
            ("Suggester", "Recommending wine pairings"),
            ("Monitor", "Tracking budget constraints")
        ]
        
        for agent, task in agents:
            print(f"  🤖 {agent}: {task}")
            await asyncio.sleep(0.5)
            
        await asyncio.sleep(1)
        print("\nJARVIS: \"I've created a complete dinner party plan:\"")
        print("  📋 Menu: Italian themed with vegetarian options")
        print("  🛒 Shopping list: 47 items, estimated cost $120")
        print("  ⏰ Timeline: Start prep at 3 PM for 7 PM dinner")
        print("  🍷 Wine pairing: Chianti and Pinot Grigio selected")
        print("  📱 Shall I send the shopping list to your phone?")
        
    async def demo_proactive_automation(self):
        """Demonstrate proactive automation"""
        print("\n\n⚡ Proactive Automation Demo")
        print("-" * 30)
        
        print("Scenario: End of workday routine detected")
        await asyncio.sleep(1)
        
        print("\nJARVIS: *automatically executing end-of-day routine*")
        automations = [
            "✅ Saving and backing up work documents",
            "📧 Drafting follow-up emails from today's meetings",
            "📅 Preparing tomorrow's calendar summary",
            "🏠 Adjusting home temperature for arrival",
            "💡 Turning on porch lights (sunset detected)",
            "🎵 Queuing relaxation playlist for drive home"
        ]
        
        for action in automations:
            print(f"  {action}")
            await asyncio.sleep(0.6)
            
        print("\nJARVIS: \"Your end-of-day routine is complete. Drive safely!\"")
        
    async def demo_continuous_learning(self):
        """Demonstrate learning capabilities"""
        print("\n\n📚 Continuous Learning Demo")
        print("-" * 30)
        
        print("Learning from recent interactions:")
        await asyncio.sleep(1)
        
        learnings = [
            {
                "pattern": "User prefers morning meetings after 9:30 AM",
                "confidence": "94%",
                "action": "Adjusted scheduling preferences"
            },
            {
                "pattern": "User asks about weather before outdoor activities",
                "confidence": "88%",
                "action": "Proactively providing weather updates"
            },
            {
                "pattern": "User orders same coffee 4/5 weekdays",
                "confidence": "97%",
                "action": "Automated coffee ordering suggestion"
            }
        ]
        
        for learning in learnings:
            print(f"\n  🧠 Learned: {learning['pattern']}")
            print(f"     Confidence: {learning['confidence']}")
            print(f"     Applied: {learning['action']}")
            await asyncio.sleep(1)
            
        print("\nJARVIS: \"I'm continuously adapting to serve you better!\"")
        
    async def demo_natural_conversation(self):
        """Demonstrate natural conversation flow"""
        print("\n\n💬 Natural Conversation Demo")
        print("-" * 30)
        
        conversation = [
            ("User", "I'm thinking about taking a vacation"),
            ("JARVIS", "That sounds wonderful! When were you thinking of going?"),
            ("User", "Maybe next month"),
            ("JARVIS", "I can help with that. Based on your preferences for warm weather and beaches, "
                      "I'd suggest considering Hawaii or the Caribbean. Your work calendar shows "
                      "the week of the 15th is relatively free."),
            ("User", "Hawaii sounds nice"),
            ("JARVIS", "Great choice! I can research flights, hotels, and activities. "
                      "I remember you enjoyed snorkeling last time - should I look for "
                      "hotels near good snorkeling spots?"),
            ("User", "Yes, and somewhere with good restaurants"),
            ("JARVIS", "I'll prioritize beachfront hotels in Maui or Oahu with highly-rated "
                      "restaurants nearby. Would you like me to create a comparison of options "
                      "with prices and availability?")
        ]
        
        for speaker, text in conversation:
            print(f"\n{speaker}: \"{text}\"")
            await asyncio.sleep(2)
            
    async def run_demo(self):
        """Run the complete demo"""
        print("\n🚀 Starting JARVIS Ultimate Demo\n")
        
        # Run each demo section
        await self.demo_voice_interaction()
        await asyncio.sleep(2)
        
        await self.demo_anticipatory_ai()
        await asyncio.sleep(2)
        
        await self.demo_swarm_intelligence()
        await asyncio.sleep(2)
        
        await self.demo_proactive_automation()
        await asyncio.sleep(2)
        
        await self.demo_continuous_learning()
        await asyncio.sleep(2)
        
        await self.demo_natural_conversation()
        
        print("\n\n✨ Demo Complete!")
        print("=" * 50)
        print("JARVIS Ultimate combines all these capabilities to create")
        print("a truly intelligent, proactive, and helpful AI assistant.")
        print("\nReady to experience it yourself? Run: ./launch_jarvis_voice_first.sh")


async def main():
    """Run the demo"""
    demo = DemoJARVIS()
    await demo.run_demo()


if __name__ == "__main__":
    print("=" * 50)
    print("     JARVIS Ultimate - Voice-First AI Assistant")
    print("=" * 50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Thank you for watching!")