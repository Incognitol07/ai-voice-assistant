package ai.picovoice.llmvoiceassistant

import ai.picovoice.porcupine.Porcupine
import ai.picovoice.porcupine.PorcupineException
import android.content.Context

class PicovoiceWakeWordEngine(private val accessKey: String) : WakeWordEngine {
    private var porcupine: Porcupine? = null
    private var wakeWordCallback: ((Int) -> Unit)? = null

    override fun start(context: Context, onWakeWordDetected: (keywordIndex: Int) -> Unit) {
        this.wakeWordCallback = onWakeWordDetected
        try {
            porcupine = Porcupine.Builder()
                .setAccessKey(accessKey)
                .setKeyword(Porcupine.BuiltInKeyword.PICOVOICE)
                .build(context)
        } catch (e: PorcupineException) {
            e.printStackTrace()
            // In a real app we might want to pass errors up, but for the POC we print stack trace.
        }
    }

    override fun process(frame: ShortArray) {
        try {
            val keywordIndex = porcupine?.process(frame) ?: -1
            if (keywordIndex == 0) {
                wakeWordCallback?.invoke(keywordIndex)
            }
        } catch (e: PorcupineException) {
            e.printStackTrace()
        }
    }

    override fun stop() {
        porcupine?.delete()
        porcupine = null
        wakeWordCallback = null
    }
}
