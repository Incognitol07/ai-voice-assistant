package ai.picovoice.llmvoiceassistant

class RTFProfiler(private val sampleRate: Int) {
    private var computeSec: Double = 0.0
    private var audioSec: Double = 0.0
    private var tickSec: Double = 0.0

    fun tick() {
        this.tickSec = System.nanoTime() / 1e9
    }

    fun tock(pcm: ShortArray?) {
        this.computeSec += (System.nanoTime() / 1e9) - this.tickSec
        if (pcm != null && pcm.isNotEmpty()) {
            this.audioSec += pcm.size / this.sampleRate.toDouble()
        }
    }

    fun rtf(): Double {
        val rtf = this.computeSec / this.audioSec
        this.computeSec = 0.0
        this.audioSec = 0.0
        return rtf
    }
}
