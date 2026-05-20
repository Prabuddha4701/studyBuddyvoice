RATE=10000
num_samples=2000
def play_beep(frequency=880, duration=0.2, volume=0.5):
    """Play a beep through the output stream."""
    num_samples = int(RATE * duration)
    t = np.linspace(0, duration, num_samples, False)
    wave = (np.sin(2 * np.pi * frequency * t) * volume * 32767).astype(np.int16)
    output_stream.write(wave.tobytes())

play_beep()