package ai.picovoice.llmvoiceassistant

import android.content.Context

interface WakeWordEngine {
    /**
     * Initializes and starts the wake word engine.
     * @param context The Android Context.
     * @param onError Callback invoked if initialization fails.
     * @param onWakeWordDetected Callback invoked when the wake word is detected. The keywordIndex is passed (0 for primary wake word).
     */
    fun start(
        context: Context,
        onError: (String) -> Unit,
        onWakeWordDetected: (keywordIndex: Int) -> Unit
    )

    /**
     * Processes a single frame of audio data (typically 512 samples at 16kHz).
     * @param frame The audio frame to process.
     */
    fun process(frame: ShortArray)

    /**
     * Stops the engine and releases any underlying resources.
     */
    fun stop()
}

