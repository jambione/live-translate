import ollama
import time
import sys

print("LLM Action processor started (every 3 seconds)...\n")

last_action_time = 0.0
LLM_INTERVAL = 3.0

try:
    for line in sys.stdin:
        line = line.strip()
        if not line.startswith("TEXT:"):
            continue

        text = line[5:].strip()
        if len(text) < 15:
            continue

        current_time = time.time()
        if current_time - last_action_time < LLM_INTERVAL:
            continue

        try:
            resp = ollama.chat(
                model="phi3:mini",
                messages=[{"role": "user", "content": f"Short realtime action based on this text only. Reply 'NO ACTION' if none needed.\nText: {text}"}],
                options={"temperature": 0.0, "num_ctx": 1024}
            )
            action = resp['message']['content'].strip()
            if action and "NO ACTION" not in action.upper():
                print(f"ACTION: {action}")
                sys.stdout.flush()
            last_action_time = current_time
        except:
            pass
except KeyboardInterrupt:
    print("\nLLM processor stopped.")