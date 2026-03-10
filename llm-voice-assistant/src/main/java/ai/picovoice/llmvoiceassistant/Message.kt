package ai.picovoice.llmvoiceassistant

import java.util.concurrent.atomic.AtomicLong

/**
 * A single turn in the conversation: user utterance or assistant response.
 * Text is mutable so streaming tokens can be appended without new object allocations.
 */
class Message(
    val role: Role,
    initialText: String
) {
    enum class Role {
        USER,
        ASSISTANT
    }

    val id: Long = ID_COUNTER.getAndIncrement()
    private val text: StringBuilder = StringBuilder(initialText)

    fun getText(): String = text.toString()

    /** Append a streaming token or transcript chunk to this message. */
    fun append(chunk: String) {
        text.append(chunk)
    }

    companion object {
        private val ID_COUNTER = AtomicLong(0)
    }
}
