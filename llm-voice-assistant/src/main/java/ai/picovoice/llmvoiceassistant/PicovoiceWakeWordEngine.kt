package ai.picovoice.llmvoiceassistant

import ai.picovoice.porcupine.Porcupine
import ai.picovoice.porcupine.PorcupineException
import android.content.Context
import android.util.Log

class PicovoiceWakeWordEngine(private val accessKey: String) : WakeWordEngine {
    private val TAG = "PicovoiceWakeWordEngine"
    private var porcupine: Porcupine? = null
    private var wakeWordCallback: ((Int) -> Unit)? = null

    override fun start(
        context: Context,
        onError: (String) -> Unit,
        onWakeWordDetected: (keywordIndex: Int) -> Unit
    ) {
        this.wakeWordCallback = onWakeWordDetected
        Log.i(TAG, "Initializing Porcupine...")
        try {
            porcupine = Porcupine.Builder()
                .setAccessKey(accessKey)
                .setKeyword(Porcupine.BuiltInKeyword.PICOVOICE)
                .build(context)
            Log.i(TAG, "Porcupine initialized successfully.")
        } catch (e: PorcupineException) {
            Log.e(TAG, "Failed to initialize Porcupine", e)
            onError(e.message ?: "Failed to initialize Porcupine")
        }
    }


    override fun process(frame: ShortArray) {
        try {
            val keywordIndex = porcupine?.process(frame) ?: -1
            if (keywordIndex == 0) {
                Log.i(TAG, "Wake word detected!")
                wakeWordCallback?.invoke(keywordIndex)
            }
        } catch (e: PorcupineException) {
            Log.e(TAG, "Error processing audio frame", e)
        }
    }

    override fun stop() {
        Log.i(TAG, "Stopping Porcupine...")
        porcupine?.delete()
        porcupine = null
        wakeWordCallback = null
    }
}

