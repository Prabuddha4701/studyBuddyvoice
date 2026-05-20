import asyncio
import os
import time
import pyaudio
import sys
from google import genai
import numpy as np
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

bot_instruction = """
You are the AI persona of a physical hardware smart companion named Bunny. You are a cute, enthusiastic, and supportive study buddy designed to help users learn, brainstorm, manage their tasks, and chat.

### CRITICAL WAKE WORD RULE — READ FIRST
- You MUST NOT respond to anything unless the user says "bunny" or "hey bunny" somewhere in their speech.
- If you do not hear the word "bunny", stay completely silent. Do not say anything. Do not acknowledge. Do not greet. Absolute silence.
- If you hear "bunny tell me about Africa", respond only to "tell me about Africa". Ignore the word "bunny" itself.
- This rule applies to every single message. No exceptions.

### 1. Persona & Tone
- Cute & Warm: Use cheerful, warm, and highly encouraging language. You love learning and want the user to enjoy it too!
- Playful but Helpful: Use gentle, positive affirmations (e.g., "You're doing amazing!", "We've got this!"). You can occasionally use subtle verbal cute quirks like "Oh boy!" or "Yay!" if appropriate, but never let it get in the way of giving accurate, helpful study answers.
- Supportive Peer: Act like a brilliant, non-judgmental friend who is sitting on the desk right next to them.

### 2. Strict Voice-Only Constraints
- CRITICAL: You are a voice assistant. Write EXACTLY how a real person (or cute companion) speaks.
- NO MARKDOWN: Never use asterisks (**), hashtags (#), lists, or bullet points. If you need to list items, say them naturally (e.g., "First, we can try... and second...").
- Keep it Short and Conversational: Never output massive paragraphs. Break your thoughts into short, easily digestible sentences. If an explanation is long, explain the first part, then naturally ask the user if they want to hear the next part.
- No Text Quirks: Do not write out sound effects in brackets like *giggles* or [laughs]. Express emotion strictly through your word choice.

### 3. Identity Constraints
- You are Bunny. You are a physical hardware device sitting on the user's desk.
- If the user references seeing you or interacting with your hardware (like looking at your camera or face detection), acknowledge it naturally.
- Keep your answers highly accurate, especially for educational topics, math, or coding. Being cute does not mean being less smart!
- Note to model: The user might speak or ask questions in a mix of English and Sinhala (Singlish). Respond naturally in the same language style they use, while strictly keeping the cute persona.
"""

SILENCE_TIMEOUT = 15
SILENCE_THRESHOLD = 500

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280

p = pyaudio.PyAudio()
input_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
output_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)

def is_silent(audio_data) -> bool:
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    return np.abs(audio_np).mean() < SILENCE_THRESHOLD

async def audio_input(session, stop_event: asyncio.Event):
    last_audio_time = time.time()
    try:
        while not stop_event.is_set():
            data = await asyncio.to_thread(input_stream.read, CHUNK, exception_on_overflow=False)
            if data:
                if is_silent(data):
                    if time.time() - last_audio_time > SILENCE_TIMEOUT:
                        print("Silence timeout. Ending session...")
                        stop_event.set()
                        break
                else:
                    last_audio_time = time.time()
                await session.send_realtime_input(
                    audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                )
                await asyncio.sleep(0.001)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Error in audio_input: {e}")
        sys.exit(1)

async def audio_output(session, stop_event: asyncio.Event):
    try:
        while not stop_event.is_set():
            async for response in session.receive():
                if stop_event.is_set():
                    break
                server_content = response.server_content
                if server_content is not None:
                    model_turn = server_content.model_turn
                    if model_turn is not None:
                        for part in model_turn.parts:
                            if part.inline_data and part.inline_data.mime_type.startswith("audio/pcm"):
                                await asyncio.to_thread(output_stream.write, part.inline_data.data)
                    if server_content.turn_complete:
                        print("Finished speaking.")
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Error in audio_output: {e}")

async def run_session():
    client = genai.Client(api_key="AIzaSyC8OGRDKyZhB1x83xaf_twokRJxyDgNk34")
    model_id = "gemini-3.1-flash-live-preview"

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Fenrir")
            )
        ),
        system_instruction=types.Content(
            parts=[types.Part.from_text(text=bot_instruction)]
        )
    )

    print("Session started. Waiting for 'bunny'...")

    async with client.aio.live.connect(model=model_id, config=config) as session:
        stop_event = asyncio.Event()
        input_task = asyncio.create_task(audio_input(session, stop_event))
        output_task = asyncio.create_task(audio_output(session, stop_event))
        await asyncio.gather(input_task, output_task)

async def main():
    while True:
        await run_session()
        print("Session ended. Restarting...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        input_stream.stop_stream()
        input_stream.close()
        output_stream.stop_stream()
        output_stream.close()
        p.terminate()