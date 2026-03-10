package ai.picovoice.llmvoiceassistant

class TPSProfiler {
    private var numTokens: Int = 0
    private var startSec: Long = 0
    private var endSec: Long = 0

    fun tock() {
        if (this.startSec == 0L) {
            this.startSec = System.nanoTime()
        } else {
            this.endSec = System.nanoTime()
            this.numTokens += 1
        }
    }

    fun tps(): Double {
        val tps = this.numTokens / ((this.endSec - this.startSec) / 1e9)
        this.numTokens = 0
        this.startSec = 0
        this.endSec = 0
        return tps
    }
}
