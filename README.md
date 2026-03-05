# LLM Voice Assistant - Android

[![LLM VA in Action](https://img.youtube.com/vi/5JkDVbkedBU/0.jpg)](https://www.youtube.com/watch?v=5JkDVbkedBU)

Wake-word triggered voice assistant. Say "Picovoice", ask something, get a spoken answer.

**Stack:**

- Wake word: Porcupine
- Speech-to-text: Cheetah
- LLM: GPT-4o via GitHub Models (cloud)
- Text-to-speech: Orca

---

## Requirements

- Android SDK 26+
- A [Picovoice Console](https://console.picovoice.ai/) account (free) → grab your AccessKey
- A GitHub personal access token with `models:read` scope (for GitHub Models API)

---

## Setup

**1. Add your keys to `local.properties`** (never commit this file):

```properties
PICOVOICE_ACCESS_KEY=your-key-here
GITHUB_TOKEN=your-github-pat-here
```

Both are read by Gradle at build time and baked into `BuildConfig`, no hardcoding needed.

**2. Drop the model files into `llm-voice-assistant/src/main/assets/`:**

| File | Where to get it |
|---|---|
| `cheetah_params.pv` | [Picovoice Console](https://console.picovoice.ai/) → Cheetah |
| `orca_params_female.pv` | [Picovoice Console](https://console.picovoice.ai/) → Orca |

**3. Build and run** - open the project in Android Studio, connect a device, hit Run.

---

## Usage

1. Tap **Start** - initialises Porcupine, Cheetah, Orca, and connects the GitHub Models client.
2. Say **"Picovoice"** to trigger STT.
3. Ask your question. Cheetah detects end-of-speech automatically.
4. The assistant responds in text and audio. Say "Picovoice" again mid-response to interrupt.
5. **Clear** resets the conversation history. **Back arrow** returns to the load screen.

---

## Custom Wake Word

Default wake phrase is `Picovoice`. To use your own:

1. Train a custom wake word on [Picovoice Console](https://console.picovoice.ai/) → Porcupine.
2. Download the `.ppn` file and put it in `src/main/assets/`.
3. Swap the builder call in `MainActivity.java`:

```java
porcupine = new Porcupine.Builder()
        .setAccessKey(ACCESS_KEY)
        .setKeywordPath("your_wake_word.ppn")
        .build(getApplicationContext());
```

---

## Profiling

Check `logcat` (tag `PICOVOICE`) after a response. Two metrics are logged:

- **RTF (Real-time Factor)** - compute time / audio length. Lower is better. Below 1.0 means faster than real-time.
- **TPS (Tokens per Second)** - LLM output speed. Higher is better.
