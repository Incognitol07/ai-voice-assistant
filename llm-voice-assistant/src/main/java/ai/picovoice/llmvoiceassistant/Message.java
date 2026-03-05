/*
    Copyright 2024 Picovoice Inc.

    You may not use this file except in compliance with the license. A copy of the license is
    located in the "LICENSE" file accompanying this source.
*/

package ai.picovoice.llmvoiceassistant;

import java.util.concurrent.atomic.AtomicLong;

/**
 * A single turn in the conversation: user utterance or assistant response.
 * Text is mutable so streaming tokens can be appended without new object allocations.
 */
public class Message {

    public enum Role {
        USER,
        ASSISTANT
    }

    private static final AtomicLong ID_COUNTER = new AtomicLong(0);

    private final long id;
    private final Role role;
    private final StringBuilder text;

    public Message(Role role, String initialText) {
        this.id   = ID_COUNTER.getAndIncrement();
        this.role = role;
        this.text = new StringBuilder(initialText);
    }

    public long getId() {
        return id;
    }

    public Role getRole() {
        return role;
    }

    public String getText() {
        return text.toString();
    }

    /** Append a streaming token or transcript chunk to this message. */
    public void append(String chunk) {
        text.append(chunk);
    }
}
