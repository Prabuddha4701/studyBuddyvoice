import speech_recognition as sr

def listen_for_command(mic_index=None):
    # Initialize the recognizer
    recognizer = sr.Recognizer()
    
    # Tweak settings to make detection faster and snappier
    recognizer.pause_threshold = 1.0  # seconds of silence before considering a phrase done
    recognizer.energy_threshold = 300  # loudness threshold for mic input
    
    print("\n🎤 Ready to capture your voice command...")
    
    try:
        # Open the specific microphone source
        with sr.Microphone(device_index=mic_index) as source:
            print("🔊 Listening for command... Speak now!")
            
            # Adjusts for background ambient noise automatically for 1 second
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Listen to input with an 8-second cutoff window
            audio = recognizer.listen(source, phrase_time_limit=8)
            print("🔄 Processing audio chunks...")
            
        # Send audio payload to Google Speech Recognition cloud API
        text = recognizer.recognize_google(audio).lower()
        print(f"🎯 Heard: \"{text}\"\n")
        return text
        
    except sr.UnknownValueError:
        print("❌ Could not understand the audio. Please try again.")
        return ""
    except sr.RequestError as e:
        print(f"❌ Could not request results from Google Speech Recognition service: {e}")
        return ""
    except Exception as e:
        print(f"❌ Unexpected error occurred: {e}")
        return ""

# --- Quick Local Testing block ---
if __name__ == "__main__":
    # If you don't know your index, leaving it as None uses the system default mic
    command = listen_for_command(mic_index=None)