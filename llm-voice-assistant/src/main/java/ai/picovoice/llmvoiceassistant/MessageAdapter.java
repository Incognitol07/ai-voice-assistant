/*
    Copyright 2024 Picovoice Inc.

    You may not use this file except in compliance with the license. A copy of the license is
    located in the "LICENSE" file accompanying this source.
*/

package ai.picovoice.llmvoiceassistant;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;
import java.util.List;

/**
 * Drives the chat bubble list.
 *
 * Two view types: USER (right-aligned blue) and ASSISTANT (left-aligned dark).
 *
 * Key design choice: {@link #appendToLast(String)} patches the last item in-place,
 * enabling smooth token-by-token streaming without visual jank from full list redraws.
 */
public class MessageAdapter extends RecyclerView.Adapter<MessageAdapter.MessageViewHolder> {

    private static final int VIEW_TYPE_USER      = 0;
    private static final int VIEW_TYPE_ASSISTANT = 1;

    private final List<Message> messages = new ArrayList<>();

    public MessageAdapter() {
        setHasStableIds(true);
    }

    // -----------------------------------------------------------------------
    // RecyclerView.Adapter contract
    // -----------------------------------------------------------------------

    @Override
    public int getItemViewType(int position) {
        return messages.get(position).getRole() == Message.Role.USER
                ? VIEW_TYPE_USER
                : VIEW_TYPE_ASSISTANT;
    }

    @NonNull
    @Override
    public MessageViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        int layoutId = viewType == VIEW_TYPE_USER
                ? R.layout.item_message_user
                : R.layout.item_message_assistant;
        View view = LayoutInflater.from(parent.getContext()).inflate(layoutId, parent, false);
        return new MessageViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull MessageViewHolder holder, int position) {
        holder.text.setText(messages.get(position).getText());
    }

    @Override
    public long getItemId(int position) {
        return messages.get(position).getId();
    }

    @Override
    public int getItemCount() {
        return messages.size();
    }

    // -----------------------------------------------------------------------
    // Public API — must be called on the main thread
    // -----------------------------------------------------------------------

    /** Append a new message bubble to the bottom of the list. */
    public void addMessage(Message message) {
        messages.add(message);
        notifyItemInserted(messages.size() - 1);
    }

    /**
     * Append a token or transcript chunk to the last message in the list.
     * Used for real-time STT transcript and streaming LLM tokens so the bubble
     * grows in-place rather than the list re-rendering from scratch.
     */
    public void appendToLast(String chunk) {
        if (messages.isEmpty()) return;
        int lastIndex = messages.size() - 1;
        messages.get(lastIndex).append(chunk);
        notifyItemChanged(lastIndex);
    }

    /** Remove all messages and reset the list. */
    public void clear() {
        int count = messages.size();
        if (count == 0) return;
        messages.clear();
        notifyItemRangeRemoved(0, count);
    }

    // -----------------------------------------------------------------------
    // ViewHolder
    // -----------------------------------------------------------------------

    static class MessageViewHolder extends RecyclerView.ViewHolder {
        final TextView text;

        MessageViewHolder(View itemView) {
            super(itemView);
            text = itemView.findViewById(R.id.messageText);
        }
    }
}
