package ai.picovoice.llmvoiceassistant

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

/**
 * Drives the chat bubble list.
 *
 * Two view types: USER (right-aligned blue) and ASSISTANT (left-aligned dark).
 *
 * Key design choice: [appendToLast] patches the last item in-place,
 * enabling smooth token-by-token streaming without visual jank from full list redraws.
 */
class MessageAdapter : RecyclerView.Adapter<MessageAdapter.MessageViewHolder>() {

    private val messages: MutableList<Message> = ArrayList()

    init {
        setHasStableIds(true)
    }

    // -----------------------------------------------------------------------
    // RecyclerView.Adapter contract
    // -----------------------------------------------------------------------

    override fun getItemViewType(position: Int): Int {
        return if (messages[position].role == Message.Role.USER)
            VIEW_TYPE_USER
        else
            VIEW_TYPE_ASSISTANT
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MessageViewHolder {
        val layoutId = if (viewType == VIEW_TYPE_USER)
            R.layout.item_message_user
        else
            R.layout.item_message_assistant
        val view = LayoutInflater.from(parent.context).inflate(layoutId, parent, false)
        return MessageViewHolder(view)
    }

    override fun onBindViewHolder(holder: MessageViewHolder, position: Int) {
        holder.text.text = messages[position].getText()
    }

    override fun getItemId(position: Int): Long {
        return messages[position].id
    }

    override fun getItemCount(): Int {
        return messages.size
    }

    // -----------------------------------------------------------------------
    // Public API — must be called on the main thread
    // -----------------------------------------------------------------------

    /** Append a new message bubble to the bottom of the list.  */
    fun addMessage(message: Message) {
        messages.add(message)
        notifyItemInserted(messages.size - 1)
    }

    /**
     * Append a token or transcript chunk to the last message in the list.
     * Used for real-time STT transcript and streaming LLM tokens so the bubble
     * grows in-place rather than the list re-rendering from scratch.
     */
    fun appendToLast(chunk: String) {
        if (messages.isEmpty()) return
        val lastIndex = messages.size - 1
        messages[lastIndex].append(chunk)
        notifyItemChanged(lastIndex)
    }

    /** Remove all messages and reset the list.  */
    fun clear() {
        val count = messages.size
        if (count == 0) return
        messages.clear()
        notifyItemRangeRemoved(0, count)
    }

    // -----------------------------------------------------------------------
    // ViewHolder
    // -----------------------------------------------------------------------

    class MessageViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val text: TextView = itemView.findViewById(R.id.messageText)
    }

    companion object {
        private const val VIEW_TYPE_USER = 0
        private const val VIEW_TYPE_ASSISTANT = 1
    }
}
