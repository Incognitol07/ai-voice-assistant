/*
    Copyright 2024 Picovoice Inc.

    You may not use this file except in compliance with the license. A copy of the license is
    located in the "LICENSE" file accompanying this source.

    Unless required by applicable law or agreed to in writing, software distributed under the
    License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
    express or implied. See the License for the specific language governing permissions and
    limitations under the License.
*/

package ai.picovoice.llmvoiceassistant;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioTrack;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.app.ActivityCompat;
import androidx.core.content.res.ResourcesCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.recyclerview.widget.SimpleItemAnimator;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

import ai.picovoice.android.voiceprocessor.VoiceProcessor;
import ai.picovoice.android.voiceprocessor.VoiceProcessorException;
import ai.picovoice.cheetah.Cheetah;
import ai.picovoice.cheetah.CheetahException;
import ai.picovoice.cheetah.CheetahTranscript;
import ai.picovoice.orca.Orca;
import ai.picovoice.orca.OrcaException;
import ai.picovoice.orca.OrcaSynthesizeParams;
import ai.picovoice.picollm.PicoLLM;
import ai.picovoice.picollm.PicoLLMCompletion;
import ai.picovoice.picollm.PicoLLMDialog;
import ai.picovoice.picollm.PicoLLMException;
import ai.picovoice.picollm.PicoLLMGenerateParams;
import ai.picovoice.porcupine.Porcupine;
import ai.picovoice.porcupine.PorcupineException;

public class MainActivity extends AppCompatActivity {

    private enum UIState {
        INIT,
        LOADING_MODEL,
        WAKE_WORD,
        STT,
        LLM_TTS
    }

    private static final String ACCESS_KEY = BuildConfig.PICOVOICE_ACCESS_KEY;

    private static final String STT_MODEL_FILE = "cheetah_params.pv";

    private static final String TTS_MODEL_FILE = "orca_params_female.pv";

    private static final String SYSTEM_PROMPT = null;

    private static final int COMPLETION_TOKEN_LIMIT = 128;

    private static final int TTS_WARMUP_SECONDS = 1;

    private static final String[] STOP_PHRASES = new String[]{
            "</s>",             // Llama-2, Mistral, and Mixtral
            "<end_of_turn>",    // Gemma
            "<|endoftext|>",    // Phi-2
            "<|eot_id|>",       // Llama-3
            "<|end|>", "<|user|>", "<|assistant|>", // Phi-3
    };

    private final VoiceProcessor voiceProcessor = VoiceProcessor.getInstance();

    private Porcupine porcupine;
    private Cheetah cheetah;
    private PicoLLM picollm;
    private Orca orca;

    private PicoLLMDialog dialog;

    private PicoLLMCompletion finalCompletion;

    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    private final ExecutorService engineExecutor = Executors.newSingleThreadExecutor();
    private final ExecutorService ttsSynthesizeExecutor = Executors.newSingleThreadExecutor();
    private final ExecutorService ttsPlaybackExecutor = Executors.newSingleThreadExecutor();

    // Accumulates LLM tokens between UI frames so we post to the main thread at
    // most once per ~16 ms (one vsync) rather than once per token. Decouples
    // inference throughput from render rate and reduces thread-scheduler pressure.
    private final StringBuilder pendingTokenBuffer = new StringBuilder();
    private Runnable flushPendingTokens;  // initialized in onCreate after messageAdapter

    private AudioTrack ttsOutput;

    private UIState currentState = UIState.INIT;

    private StringBuilder llmPromptText = new StringBuilder();

    private ConstraintLayout loadModelLayout;
    private ConstraintLayout chatLayout;

    private Button loadModelButton;
    private TextView loadModelText;
    private ProgressBar loadModelProgress;

    private RecyclerView chatRecyclerView;
    private MessageAdapter messageAdapter;

    private TextView statusText;

    private VoiceStateView voiceStateView;

    private ImageButton loadNewModelButton;

    private ImageButton clearTextButton;

    @SuppressLint({"DefaultLocale", "SetTextI18n"})
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main_layout);

        loadModelLayout = findViewById(R.id.loadModelLayout);
        chatLayout = findViewById(R.id.chatLayout);

        loadModelText = findViewById(R.id.loadModelText);
        loadModelProgress = findViewById(R.id.loadModelProgress);
        loadModelButton = findViewById(R.id.loadModelButton);

        loadModelButton.setOnClickListener(view -> {
            modelSelection.launch(new String[]{"application/octet-stream"});
        });

        updateUIState(UIState.INIT);

        messageAdapter = new MessageAdapter();
        flushPendingTokens = () -> {
            final String chunk;
            synchronized (pendingTokenBuffer) {
                chunk = pendingTokenBuffer.toString();
                pendingTokenBuffer.setLength(0);
            }
            if (!chunk.isEmpty()) {
                messageAdapter.appendToLast(chunk);
                scrollToBottom();
            }
        };
        chatRecyclerView = findViewById(R.id.chatRecyclerView);
        LinearLayoutManager layoutManager = new LinearLayoutManager(this);
        layoutManager.setStackFromEnd(true);
        chatRecyclerView.setLayoutManager(layoutManager);
        chatRecyclerView.setAdapter(messageAdapter);
        // Disable the default cross-fade animation on item updates. Without this
        // every notifyItemChanged call (i.e. every streamed token) triggers a
        // brief fade-out/fade-in on the bubble, producing the jarring flicker.
        ((SimpleItemAnimator) chatRecyclerView.getItemAnimator())
                .setSupportsChangeAnimations(false);

        statusText = findViewById(R.id.statusText);
        voiceStateView = findViewById(R.id.voiceStateView);

        loadNewModelButton = findViewById(R.id.loadNewModelButton);
        loadNewModelButton.setOnClickListener(view -> {
            if (picollm != null) {
                picollm.delete();
                picollm = null;
            }
            updateUIState(UIState.INIT);
            mainHandler.post(() -> messageAdapter.clear());
        });

        clearTextButton = findViewById(R.id.clearButton);
        clearTextButton.setOnClickListener(view -> {
            // Submit the dialog reset to the engine executor so it is serialized
            // with any in-flight addHumanRequest / addLLMResponse calls, eliminating
            // the race between the main thread and the inference thread.
            engineExecutor.submit(() -> {
                try {
                    dialog = picollm.getDialogBuilder().setSystem(SYSTEM_PROMPT).build();
                } catch (PicoLLMException e) {
                    updateUIState(UIState.WAKE_WORD);
                    mainHandler.post(() -> messageAdapter.addMessage(
                            new Message(Message.Role.ASSISTANT, e.toString())));
                }
            });
            mainHandler.post(() -> {
                messageAdapter.clear();
                clearTextButton.setEnabled(false);
                clearTextButton.setImageDrawable(
                        ResourcesCompat.getDrawable(getResources(),
                                R.drawable.clear_button_disabled,
                                null));
            });
        });
    }

    ActivityResultLauncher<String[]> modelSelection = registerForActivityResult(
            new ActivityResultContracts.OpenDocument(),
            new ActivityResultCallback<Uri>() {
                @SuppressLint("SetTextI18n")
                @Override
                public void onActivityResult(Uri selectedUri) {
                    updateUIState(UIState.LOADING_MODEL);

                    if (selectedUri == null) {
                        updateUIState(UIState.INIT);
                        mainHandler.post(() -> loadModelText.setText("No file selected"));
                        return;
                    }

                    engineExecutor.submit(() -> {
                        File llmModelFile = extractModelFile(selectedUri);
                        if (llmModelFile == null || !llmModelFile.exists()) {
                            updateUIState(UIState.INIT);
                            mainHandler.post(() -> loadModelText.setText("Unable to access selected file"));
                            return;
                        }

                        initEngines(llmModelFile);
                    });
                }
            });

    private void initEngines(File modelFile) {
        mainHandler.post(() -> loadModelText.setText("Loading Porcupine..."));
        try {
            porcupine = new Porcupine.Builder()
                    .setAccessKey(ACCESS_KEY)
                    .setKeyword(Porcupine.BuiltInKeyword.PICOVOICE)
                    .build(getApplicationContext());
        } catch (PorcupineException e) {
            onEngineInitError(e.getMessage());
            return;
        }

        mainHandler.post(() -> loadModelText.setText("Loading Cheetah..."));
        try {
            cheetah = new Cheetah.Builder()
                    .setAccessKey(ACCESS_KEY)
                    .setModelPath(STT_MODEL_FILE)
                    .setEnableAutomaticPunctuation(true)
                    .build(getApplicationContext());
        } catch (CheetahException e) {
            onEngineInitError(e.getMessage());
            return;
        }

        mainHandler.post(() -> loadModelText.setText("Loading picoLLM..."));
        try {
            picollm = new PicoLLM.Builder()
                    .setAccessKey(ACCESS_KEY)
                    .setModelPath(modelFile.getAbsolutePath())
                    .build();
            dialog = picollm.getDialogBuilder().setSystem(SYSTEM_PROMPT).build();
        } catch (PicoLLMException e) {
            onEngineInitError(e.getMessage());
            return;
        }

        mainHandler.post(() -> loadModelText.setText("Loading Orca..."));
        try {
            orca = new Orca.Builder()
                    .setAccessKey(ACCESS_KEY)
                    .setModelPath(TTS_MODEL_FILE)
                    .build(getApplicationContext());
        } catch (OrcaException e) {
            onEngineInitError(e.getMessage());
            return;
        }

        updateUIState(UIState.WAKE_WORD);

        voiceProcessor.addFrameListener(this::runWakeWordSTT);

        voiceProcessor.addErrorListener(error -> {
            onEngineProcessError(error.getMessage());
        });

        startWakeWordListening();
    }

    private void runWakeWordSTT(short[] frame) {
        if (currentState == UIState.WAKE_WORD) {
            try {
                int keywordIndex = porcupine.process(frame);
                if (keywordIndex == 0) {
                    interrupt();

                    llmPromptText = new StringBuilder();
                    updateUIState(UIState.STT);
                }
            } catch (PorcupineException e) {
                onEngineProcessError(e.getMessage());
            }
        } else if (currentState == UIState.LLM_TTS) {
            try {
                int keywordIndex = porcupine.process(frame);
                if (keywordIndex == 0) {
                    interrupt();
                }
            } catch (PorcupineException e) {
                onEngineProcessError(e.getMessage());
            }
        } else if (currentState == UIState.STT) {
            try {
                CheetahTranscript result = cheetah.process(frame);
                llmPromptText.append(result.getTranscript());
                mainHandler.post(() -> {
                    messageAdapter.appendToLast(result.getTranscript());
                    scrollToBottom();
                });

                if (result.getIsEndpoint()) {
                    CheetahTranscript finalResult = cheetah.flush();
                    llmPromptText.append(finalResult.getTranscript());
                    mainHandler.post(() -> {
                        messageAdapter.appendToLast(finalResult.getTranscript());
                        scrollToBottom();
                    });

                    runLLM(llmPromptText.toString());
                }
            } catch (CheetahException e) {
                onEngineProcessError(e.getMessage());
            }
        }
    }

    private void runLLM(String prompt) {
        if (prompt.length() == 0) {
            return;
        }

        AtomicBoolean isQueueingTokens = new AtomicBoolean(false);
        CountDownLatch tokensReadyLatch = new CountDownLatch(1);
        ConcurrentLinkedQueue<String> tokenQueue = new ConcurrentLinkedQueue<>();

        AtomicBoolean isQueueingPcm = new AtomicBoolean(false);
        CountDownLatch pcmReadyLatch = new CountDownLatch(1);
        ConcurrentLinkedQueue<short[]> pcmQueue = new ConcurrentLinkedQueue<>();

        updateUIState(UIState.LLM_TTS);

        finalCompletion = null;

        mainHandler.post(() -> messageAdapter.addMessage(new Message(Message.Role.ASSISTANT, "")));

        engineExecutor.submit(() -> {
            TPSProfiler picoLLMProfiler = new TPSProfiler();
            try {
                isQueueingTokens.set(true);

                dialog.addHumanRequest(prompt);
                finalCompletion = picollm.generate(
                        dialog.getPrompt(),
                        new PicoLLMGenerateParams.Builder()
                                .setStreamCallback(token -> {
                                    picoLLMProfiler.tock();
                                    if (token != null && token.length() > 0) {
                                        boolean containsStopPhrase = false;
                                        for (String k : STOP_PHRASES) {
                                            if (token.contains(k)) {
                                                containsStopPhrase = true;
                                                break;
                                            }
                                        }

                                        if (!containsStopPhrase && currentState == UIState.LLM_TTS) {
                                            tokenQueue.add(token);
                                            tokensReadyLatch.countDown();

                                            // Batch: accumulate token, schedule a single
                                            // UI flush no more than once per frame (~16ms).
                                            synchronized (pendingTokenBuffer) {
                                                pendingTokenBuffer.append(token);
                                            }
                                            mainHandler.removeCallbacks(flushPendingTokens);
                                            mainHandler.postDelayed(flushPendingTokens, 16);
                                        }
                                    }
                                })
                                .setCompletionTokenLimit(COMPLETION_TOKEN_LIMIT)
                                .setStopPhrases(STOP_PHRASES)
                                .build());
                dialog.addLLMResponse(finalCompletion.getCompletion());
                Log.i("PICOVOICE", String.format("TPS: %.2f", picoLLMProfiler.tps()));

                isQueueingTokens.set(false);

                // Flush any tokens that accumulated in the batch buffer during
                // the last 16ms window — ensures the final words always appear.
                mainHandler.removeCallbacks(flushPendingTokens);
                mainHandler.post(flushPendingTokens);

                mainHandler.post(() -> {
                    clearTextButton.setEnabled(true);
                    clearTextButton.setImageDrawable(
                            ResourcesCompat.getDrawable(getResources(),
                                    R.drawable.clear_button,
                                    null));
                });
                // State transition intentionally deferred to ttsPlaybackExecutor
                // so the animation persists until the last audio sample plays.

            } catch (PicoLLMException e) {
                onEngineProcessError(e.getMessage());
            }
        });

        ttsSynthesizeExecutor.submit(() -> {
            Orca.OrcaStream orcaStream;
            try {
                orcaStream = orca.streamOpen(new OrcaSynthesizeParams.Builder().build());
            } catch (OrcaException e) {
                onEngineProcessError(e.getMessage());
                return;
            }

            RTFProfiler orcaProfiler = new RTFProfiler(orca.getSampleRate());

            short[] warmupPcm;
            if (TTS_WARMUP_SECONDS > 0) {
                warmupPcm = new short[0];
            }

            try {
                tokensReadyLatch.await();
            } catch (InterruptedException e) {
                onEngineProcessError(e.getMessage());
                return;
            }

            isQueueingPcm.set(true);
            while (isQueueingTokens.get() || !tokenQueue.isEmpty()) {
                String token = tokenQueue.poll();
                if (token != null && token.length() > 0) {
                    try {
                        orcaProfiler.tick();
                        short[] pcm = orcaStream.synthesize(token);
                        orcaProfiler.tock(pcm);

                        if (pcm != null && pcm.length > 0) {
                            if (warmupPcm != null) {
                                int offset = warmupPcm.length;
                                warmupPcm = Arrays.copyOf(warmupPcm, offset + pcm.length);
                                System.arraycopy(pcm, 0, warmupPcm, offset, pcm.length);
                                if (warmupPcm.length > TTS_WARMUP_SECONDS * orca.getSampleRate()) {
                                    pcmQueue.add(warmupPcm);
                                    pcmReadyLatch.countDown();
                                    warmupPcm = null;
                                }
                            } else {
                                pcmQueue.add(pcm);
                                pcmReadyLatch.countDown();
                            }
                        }
                    } catch (OrcaException e) {
                        onEngineProcessError(e.getMessage());
                        return;
                    }
                } else {
                    // Queue is momentarily empty while the LLM is still generating.
                    // Yield the CPU for 1 ms so the inference thread isn't starved.
                    try { Thread.sleep(1); } catch (InterruptedException ignored) {}
                }
            }

            try {
                orcaProfiler.tick();
                short[] flushedPcm = orcaStream.flush();
                orcaProfiler.tock(flushedPcm);

                if (flushedPcm != null && flushedPcm.length > 0) {
                    if (warmupPcm != null) {
                        int offset = warmupPcm.length;
                        warmupPcm = Arrays.copyOf(warmupPcm, offset + flushedPcm.length);
                        System.arraycopy(flushedPcm, 0, warmupPcm, offset, flushedPcm.length);
                        pcmQueue.add(warmupPcm);
                        pcmReadyLatch.countDown();
                    }
                    else {
                        pcmQueue.add(flushedPcm);
                        pcmReadyLatch.countDown();
                    }
                }
                Log.i("PICOVOICE", String.format("RTF: %.2f", orcaProfiler.rtf()));
            } catch (OrcaException e) {
                onEngineProcessError(e.getMessage());
            }

            isQueueingPcm.set(false);

            orcaStream.close();
        });

        ttsPlaybackExecutor.submit(() -> {
            try {
                AudioAttributes audioAttributes = new AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                        .build();

                AudioFormat audioFormat = new AudioFormat.Builder()
                        .setSampleRate(orca.getSampleRate())
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build();

                ttsOutput = new AudioTrack(
                        audioAttributes,
                        audioFormat,
                        // 4× the minimum avoids underruns and reduces the frequency
                        // at which the OS must wake up the playback thread.
                        AudioTrack.getMinBufferSize(
                                orca.getSampleRate(),
                                AudioFormat.CHANNEL_OUT_MONO,
                                AudioFormat.ENCODING_PCM_16BIT) * 4,
                        AudioTrack.MODE_STREAM,
                        0);

                ttsOutput.play();
            } catch (Exception e) {
                onEngineProcessError(e.getMessage());
                return;
            }

            try {
                pcmReadyLatch.await();
            } catch (InterruptedException e) {
                onEngineProcessError(e.getMessage());
                return;
            }

            while (isQueueingPcm.get() || !pcmQueue.isEmpty()) {
                short[] pcm = pcmQueue.poll();
                if (pcm != null && pcm.length > 0 && ttsOutput.getPlayState() == AudioTrack.PLAYSTATE_PLAYING) {
                    ttsOutput.write(pcm, 0, pcm.length);
                }
            }

            if (ttsOutput.getPlayState() == AudioTrack.PLAYSTATE_PLAYING) {
                ttsOutput.flush();
                ttsOutput.stop();
            }
            ttsOutput.release();

            // Transition only after the last audio sample has been played.
            if (finalCompletion != null
                    && finalCompletion.getEndpoint() == PicoLLMCompletion.Endpoint.INTERRUPTED) {
                llmPromptText = new StringBuilder();
                updateUIState(UIState.STT);
            } else {
                updateUIState(UIState.WAKE_WORD);
            }
        });
    }

    private void interrupt() {
        try {
            picollm.interrupt();
            if (ttsOutput != null && ttsOutput.getPlayState() == AudioTrack.PLAYSTATE_PLAYING) {
                ttsOutput.stop();
            }
        } catch (PicoLLMException e) {
            onEngineProcessError(e.getMessage());
        }
    }

    private File extractModelFile(Uri uri) {
        File modelFile = new File(getApplicationContext().getFilesDir(), "model.pllm");

        try (InputStream is = getContentResolver().openInputStream(uri);
                OutputStream os = new FileOutputStream(modelFile)) {
            byte[] buffer = new byte[8192];
            int numBytesRead;
            while ((numBytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, numBytesRead);
            }
        } catch (IOException e) {
            return null;
        }

        return modelFile;
    }

    private void onEngineInitError(String message) {
        updateUIState(UIState.INIT);
        mainHandler.post(() -> loadModelText.setText(message));
    }

    private void onEngineProcessError(String message) {
        updateUIState(UIState.WAKE_WORD);
        mainHandler.post(() -> messageAdapter.addMessage(new Message(Message.Role.ASSISTANT, message)));
    }

    private void scrollToBottom() {
        int count = messageAdapter.getItemCount();
        if (count > 0) {
            chatRecyclerView.scrollToPosition(count - 1);
        }
    }

    private void startWakeWordListening() {
        if (voiceProcessor.hasRecordAudioPermission(this)) {
            try {
                voiceProcessor.start(cheetah.getFrameLength(), cheetah.getSampleRate());
            } catch (VoiceProcessorException e) {
                onEngineProcessError(e.getMessage());
            }
        } else {
            requestRecordPermission();
        }
    }

    private void requestRecordPermission() {
        ActivityCompat.requestPermissions(
                this,
                new String[]{Manifest.permission.RECORD_AUDIO},
                0);
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode,
            @NonNull String[] permissions,
            @NonNull int[] grantResults
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults.length == 0 || grantResults[0] == PackageManager.PERMISSION_DENIED) {
            onEngineProcessError("Recording permission not granted");
        } else {
            startWakeWordListening();
        }
    }

    private void updateUIState(UIState state) {
        mainHandler.post(() -> {
            switch (state) {
                case INIT:
                    loadModelLayout.setVisibility(View.VISIBLE);
                    chatLayout.setVisibility(View.INVISIBLE);
                    loadModelButton.setEnabled(true);
                    loadModelButton.setBackground(
                            ResourcesCompat.getDrawable(
                                    getResources(),
                                    R.drawable.button_background,
                                    null));
                    loadModelProgress.setVisibility(View.INVISIBLE);
                    loadModelText.setText(getResources().getString(R.string.intro_text));
                    break;
                case LOADING_MODEL:
                    loadModelLayout.setVisibility(View.VISIBLE);
                    chatLayout.setVisibility(View.INVISIBLE);
                    loadModelButton.setEnabled(false);
                    loadModelButton.setBackground(
                            ResourcesCompat.getDrawable(
                                    getResources(),
                                    R.drawable.button_disabled,
                                    null));
                    loadModelProgress.setVisibility(View.VISIBLE);
                    loadModelText.setText("Loading model...");
                    break;
                case WAKE_WORD:
                    loadModelLayout.setVisibility(View.INVISIBLE);
                    chatLayout.setVisibility(View.VISIBLE);

                    loadNewModelButton.setImageDrawable(
                            ResourcesCompat.getDrawable(
                                    getResources(),
                                    R.drawable.arrow_back_button,
                                    null));
                    loadNewModelButton.setEnabled(true);
                    voiceStateView.setState(VoiceStateView.State.WAKE_WORD);
                    statusText.setText("Say 'Picovoice'!");
                    if (messageAdapter.getItemCount() > 0) {
                        clearTextButton.setEnabled(true);
                        clearTextButton.setImageDrawable(
                                ResourcesCompat.getDrawable(getResources(),
                                        R.drawable.clear_button,
                                        null));
                    } else {
                        clearTextButton.setEnabled(false);
                        clearTextButton.setImageDrawable(
                                ResourcesCompat.getDrawable(
                                        getResources(),
                                        R.drawable.clear_button_disabled,
                                        null));
                    }
                    break;
                case STT:
                    loadModelLayout.setVisibility(View.INVISIBLE);
                    chatLayout.setVisibility(View.VISIBLE);

                    loadNewModelButton.setImageDrawable(
                            ResourcesCompat.getDrawable(
                                    getResources(),
                                    R.drawable.arrow_back_button_disabled,
                                    null));
                    loadNewModelButton.setEnabled(false);
                    voiceStateView.setState(VoiceStateView.State.STT);
                    statusText.setText("Listening...");

                    messageAdapter.addMessage(new Message(Message.Role.USER, ""));

                    clearTextButton.setEnabled(true);
                    clearTextButton.setImageDrawable(
                            ResourcesCompat.getDrawable(getResources(),
                                    R.drawable.clear_button,
                                    null));
                    break;
                case LLM_TTS:
                    loadModelLayout.setVisibility(View.INVISIBLE);
                    chatLayout.setVisibility(View.VISIBLE);

                    loadNewModelButton.setImageDrawable(
                            ResourcesCompat.getDrawable(
                                    getResources(),
                                    R.drawable.arrow_back_button_disabled,
                                    null));
                    loadNewModelButton.setEnabled(false);
                    voiceStateView.setState(VoiceStateView.State.LLM_TTS);
                    statusText.setText("Say 'Picovoice' to interrupt");
                    clearTextButton.setEnabled(false);
                    clearTextButton.setImageDrawable(
                            ResourcesCompat.getDrawable(
                                    getResources(),
                                    R.drawable.clear_button_disabled,
                                    null));
                    break;
                default:
                    break;
            }

            currentState = state;
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        engineExecutor.shutdownNow();
        ttsSynthesizeExecutor.shutdownNow();
        ttsPlaybackExecutor.shutdownNow();

        if (porcupine != null) {
            porcupine.delete();
            porcupine = null;
        }

        if (cheetah != null) {
            cheetah.delete();
            cheetah = null;
        }

        if (picollm != null) {
            picollm.delete();
            picollm = null;
        }

        if (orca != null) {
            orca.delete();
            orca = null;
        }

        if (voiceProcessor != null) {
            voiceProcessor.clearFrameListeners();
            voiceProcessor.clearErrorListeners();
        }
    }
}
