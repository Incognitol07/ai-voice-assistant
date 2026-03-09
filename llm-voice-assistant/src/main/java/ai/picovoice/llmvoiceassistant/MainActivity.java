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
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.app.ActivityCompat;
import androidx.core.content.res.ResourcesCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.recyclerview.widget.SimpleItemAnimator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
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
import ai.picovoice.porcupine.Porcupine;
import ai.picovoice.porcupine.PorcupineException;

import com.azure.ai.inference.ChatCompletionsClient;
import com.azure.ai.inference.ChatCompletionsClientBuilder;
import com.azure.ai.inference.models.ChatCompletionsOptions;
import com.azure.ai.inference.models.ChatRequestAssistantMessage;
import com.azure.ai.inference.models.ChatRequestMessage;
import com.azure.ai.inference.models.ChatRequestSystemMessage;
import com.azure.ai.inference.models.ChatRequestUserMessage;
import com.azure.ai.inference.models.StreamingChatCompletionsUpdate;
import com.azure.ai.inference.models.StreamingChatResponseMessageUpdate;
import com.azure.core.credential.AccessToken;
import com.azure.core.credential.TokenCredential;
import com.azure.core.util.CoreUtils;
import com.azure.core.util.IterableStream;

import java.time.OffsetDateTime;

import reactor.core.publisher.Mono;

import android.net.Uri;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;

import ai.picovoice.picollm.PicoLLM;
import ai.picovoice.picollm.PicoLLMCompletion;
import ai.picovoice.picollm.PicoLLMDialog;
import ai.picovoice.picollm.PicoLLMException;
import ai.picovoice.picollm.PicoLLMGenerateParams;

public class MainActivity extends AppCompatActivity {

    private enum UIState {
        INIT,
        LOADING_MODEL,
        WAKE_WORD,
        STT,
        LLM_TTS
    }

    private enum Mode {
        ON_DEVICE,
        CLOUD
    }

    private static final String ACCESS_KEY = BuildConfig.PICOVOICE_ACCESS_KEY;

    private static final String GITHUB_TOKEN = BuildConfig.GITHUB_TOKEN;

    private static final String OPENAI_ENDPOINT = "https://models.github.ai/inference";

    private static final String OPENAI_MODEL = "openai/gpt-4o";

    private static final String STT_MODEL_FILE = "cheetah_params.pv";

    private static final String TTS_MODEL_FILE = "orca_params_female.pv";

    private static final String SYSTEM_PROMPT = "You are a voice assistant. Follow these rules strictly: "
            + "1. Keep all responses under 3 sentences unless the user explicitly asks for more detail. "
            + "2. Never use markdown, bullet points, numbered lists, or special characters — your response will be spoken aloud. "
            + "3. Speak naturally and conversationally, as if talking to a person. "
            + "4. If you don't know something, say so briefly. Never make up facts. "
            + "5. For simple questions, give a single direct sentence.";

    private static final int TTS_WARMUP_SECONDS = 1;

    private static final int COMPLETION_TOKEN_LIMIT = 128;

    private static final String[] STOP_PHRASES = new String[]{
            "</s>",
            "<end_of_turn>",
            "<|endoftext|>",
            "<|eot_id|>",
            "<|end|>", "<|user|>", "<|assistant|>",
    };

    private final VoiceProcessor voiceProcessor = VoiceProcessor.getInstance();

    private Porcupine porcupine;
    private Cheetah cheetah;
    private ChatCompletionsClient chatClient;
    private PicoLLM picollm;
    private PicoLLMDialog dialog;
    private PicoLLMCompletion finalCompletion;
    private Orca orca;

    private Mode selectedMode = Mode.CLOUD;

    private List<ChatRequestMessage> conversationHistory = new ArrayList<>();

    private final AtomicBoolean interruptLLM = new AtomicBoolean(false);
    private final AtomicBoolean wasInterrupted = new AtomicBoolean(false);

    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    private final ExecutorService engineExecutor = Executors.newSingleThreadExecutor();
    private final ExecutorService ttsSynthesizeExecutor = Executors.newSingleThreadExecutor();
    private final ExecutorService ttsPlaybackExecutor = Executors.newSingleThreadExecutor();

    // batches tokens so main thread is posted at most once per frame (~16ms)
    private final StringBuilder pendingTokenBuffer = new StringBuilder();
    private Runnable flushPendingTokens;  // set up in onCreate, after messageAdapter exists

    private AudioTrack ttsOutput;

    private UIState currentState = UIState.INIT;

    private StringBuilder llmPromptText = new StringBuilder();

    private ConstraintLayout loadModelLayout;
    private ConstraintLayout chatLayout;

    private Button loadModelButton;
    private Button cloudModelButton;
    private TextView loadModelText;
    private ProgressBar loadModelProgress;

    private RecyclerView chatRecyclerView;
    private MessageAdapter messageAdapter;

    private TextView statusText;

    private VoiceStateView voiceStateView;

    private ImageButton loadNewModelButton;

    private ImageButton clearTextButton;

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

                        initEnginesOnDevice(llmModelFile);
                    });
                }
            });

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
            selectedMode = Mode.ON_DEVICE;
            modelSelection.launch(new String[]{"application/octet-stream"});
        });

        cloudModelButton = findViewById(R.id.cloudModelButton);
        cloudModelButton.setOnClickListener(view -> {
            selectedMode = Mode.CLOUD;
            updateUIState(UIState.LOADING_MODEL);
            engineExecutor.submit(() -> initEnginesCloud());
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
        // cross-fade on item updates causes flicker when streaming tokens
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
            conversationHistory.clear();
            updateUIState(UIState.INIT);
            mainHandler.post(() -> messageAdapter.clear());
        });

        clearTextButton = findViewById(R.id.clearButton);
        clearTextButton.setOnClickListener(view -> {
            // run on the engine thread so it doesn't race with an in-flight response
            engineExecutor.submit(() -> {
                if (selectedMode == Mode.ON_DEVICE) {
                    try {
                        dialog = picollm.getDialogBuilder().setSystem(SYSTEM_PROMPT).build();
                    } catch (PicoLLMException e) {
                        updateUIState(UIState.WAKE_WORD);
                        mainHandler.post(() -> messageAdapter.addMessage(
                                new Message(Message.Role.ASSISTANT, e.toString())));
                    }
                } else {
                    conversationHistory.clear();
                    if (SYSTEM_PROMPT != null) {
                        conversationHistory.add(new ChatRequestSystemMessage(SYSTEM_PROMPT));
                    }
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

    private void initEnginesCloud() {
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

        mainHandler.post(() -> loadModelText.setText("Initializing cloud LLM client..."));
        TokenCredential bearerCredential =
                tokenRequestContext -> Mono.just(
                        new AccessToken(GITHUB_TOKEN, OffsetDateTime.now().plusYears(1)));
        chatClient = new ChatCompletionsClientBuilder()
                .credential(bearerCredential)
                .endpoint(OPENAI_ENDPOINT)
                .buildClient();
        conversationHistory.clear();
        if (SYSTEM_PROMPT != null) {
            conversationHistory.add(new ChatRequestSystemMessage(SYSTEM_PROMPT));
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

    private void initEnginesOnDevice(File modelFile) {
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

        mainHandler.post(() -> messageAdapter.addMessage(new Message(Message.Role.ASSISTANT, "")));

        if (selectedMode == Mode.ON_DEVICE) {
            engineExecutor.submit(() -> {
                TPSProfiler picoLLMProfiler = new TPSProfiler();
                try {
                    isQueueingTokens.set(true);
                    wasInterrupted.set(false);

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
                    if (finalCompletion.getEndpoint() == PicoLLMCompletion.Endpoint.INTERRUPTED) {
                        wasInterrupted.set(true);
                    }
                    Log.i("PICOVOICE", String.format("TPS: %.2f", picoLLMProfiler.tps()));

                    isQueueingTokens.set(false);

                    mainHandler.removeCallbacks(flushPendingTokens);
                    mainHandler.post(flushPendingTokens);

                    mainHandler.post(() -> {
                        clearTextButton.setEnabled(true);
                        clearTextButton.setImageDrawable(
                                ResourcesCompat.getDrawable(getResources(),
                                        R.drawable.clear_button,
                                        null));
                    });
                } catch (PicoLLMException e) {
                    onEngineProcessError(e.getMessage());
                }
            });
        } else {
            engineExecutor.submit(() -> {
                try {
                    isQueueingTokens.set(true);
                    interruptLLM.set(false);
                    wasInterrupted.set(false);

                    conversationHistory.add(new ChatRequestUserMessage(prompt));
                    ChatCompletionsOptions opts = new ChatCompletionsOptions(
                            new ArrayList<>(conversationHistory));
                    opts.setModel(OPENAI_MODEL);

                    IterableStream<StreamingChatCompletionsUpdate> stream =
                            chatClient.completeStream(opts);

                    StringBuilder fullResponse = new StringBuilder();
                    for (StreamingChatCompletionsUpdate update : stream) {
                        if (interruptLLM.get()) {
                            wasInterrupted.set(true);
                            break;
                        }
                        if (CoreUtils.isNullOrEmpty(update.getChoices())) {
                            continue;
                        }
                        StreamingChatResponseMessageUpdate delta = update.getChoices().get(0).getDelta();
                        if (delta != null && delta.getContent() != null) {
                            String token = delta.getContent();
                            if (token.length() > 0 && currentState == UIState.LLM_TTS) {
                                tokenQueue.add(token);
                                tokensReadyLatch.countDown();
                                fullResponse.append(token);

                                synchronized (pendingTokenBuffer) {
                                    pendingTokenBuffer.append(token);
                                }
                                mainHandler.removeCallbacks(flushPendingTokens);
                                mainHandler.postDelayed(flushPendingTokens, 16);
                            }
                        }
                    }

                    conversationHistory.add(new ChatRequestAssistantMessage(fullResponse.toString()));

                    isQueueingTokens.set(false);

                    mainHandler.removeCallbacks(flushPendingTokens);
                    mainHandler.post(flushPendingTokens);

                    mainHandler.post(() -> {
                        clearTextButton.setEnabled(true);
                        clearTextButton.setImageDrawable(
                                ResourcesCompat.getDrawable(getResources(),
                                        R.drawable.clear_button,
                                        null));
                    });
                } catch (Exception e) {
                    onEngineProcessError(e.getMessage());
                }
            });
        }

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
                    // LLM still running, yield so we don't spin-wait
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
                        // larger buffer = fewer underruns
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

            if (wasInterrupted.get()) {
                llmPromptText = new StringBuilder();
                updateUIState(UIState.STT);
            } else {
                updateUIState(UIState.WAKE_WORD);
            }
        });
    }

    private void interrupt() {
        if (selectedMode == Mode.ON_DEVICE) {
            try {
                picollm.interrupt();
            } catch (PicoLLMException e) {
                onEngineProcessError(e.getMessage());
            }
        } else {
            interruptLLM.set(true);
        }
        if (ttsOutput != null && ttsOutput.getPlayState() == AudioTrack.PLAYSTATE_PLAYING) {
            ttsOutput.stop();
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
                    cloudModelButton.setEnabled(true);
                    cloudModelButton.setBackground(
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
                    cloudModelButton.setEnabled(false);
                    cloudModelButton.setBackground(
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
