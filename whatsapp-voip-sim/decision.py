import random

def check_voice(audio):
    fake_prob = random.uniform(0, 1)

    if fake_prob > 0.7:
        return ("FAKE", fake_prob)
    else:
        return ("REAL", fake_prob)
