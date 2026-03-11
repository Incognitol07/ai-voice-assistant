package ai.picovoice.llmvoiceassistant

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageButton
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.core.app.ActivityCompat
import androidx.core.content.res.ResourcesCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import androidx.recyclerview.widget.SimpleItemAnimator
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.time.OffsetDateTime
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import reactor.core.publisher.Mono
import ai.picovoice.android.voiceprocessor.VoiceProcessor
import ai.picovoice.android.voiceprocessor.VoiceProcessorException
import ai.picovoice.cheetah.Cheetah
import ai.picovoice.cheetah.CheetahException
import ai.picovoice.orca.Orca
import ai.picovoice.orca.OrcaException
import ai.picovoice.orca.OrcaSynthesizeParams
import ai.picovoice.picollm.PicoLLM
import ai.picovoice.picollm.PicoLLMCompletion
import ai.picovoice.picollm.PicoLLMDialog
import ai.picovoice.picollm.PicoLLMException
import ai.picovoice.picollm.PicoLLMGenerateParams
import com.azure.ai.inference.ChatCompletionsClient
import com.azure.ai.inference.ChatCompletionsClientBuilder
import com.azure.ai.inference.models.ChatCompletionsOptions
import com.azure.ai.inference.models.ChatRequestAssistantMessage
import com.azure.ai.inference.models.ChatRequestMessage
import com.azure.ai.inference.models.ChatRequestSystemMessage
import com.azure.ai.inference.models.ChatRequestUserMessage
import com.azure.core.credential.AccessToken
import com.azure.core.util.CoreUtils

class MainActivity : AppCompatActivity() {

    private enum class UIState {
        INIT, LOADING_MODEL, WAKE_WORD, STT, LLM_TTS
    }

    private enum class Mode {
        ON_DEVICE, CLOUD
    }

    private val voiceProcessor = VoiceProcessor.getInstance()
    private var wakeWordEngine: WakeWordEngine? = null
    private var cheetah: Cheetah? = null
    private var chatClient: com.azure.ai.inference.ChatCompletionsClient? = null
    private var picollm: ai.picovoice.picollm.PicoLLM? = null
    private var dialog: ai.picovoice.picollm.PicoLLMDialog? = null
    private var finalCompletion: ai.picovoice.picollm.PicoLLMCompletion? = null
    private var orca: Orca? = null
    private val bearerCredential = com.azure.core.credential.TokenCredential { _ ->
        com.azure.core.credential.AccessToken(GITHUB_TOKEN, OffsetDateTime.now().plusYears(1)).let { Mono.just(it) }
    }

    private enum class WakeWordProvider {
        PICOVOICE, OPENWAKEWORD
    }

    private var selectedMode = Mode.CLOUD
    private var selectedWakeWordProvider = WakeWordProvider.PICOVOICE
    private val conversationHistory = ArrayList<ChatRequestMessage>()

    private val interruptLLM = AtomicBoolean(false)
    private val wasInterrupted = AtomicBoolean(false)

    private val mainHandler = Handler(Looper.getMainLooper())
    private val engineExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val ttsSynthesizeExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val ttsPlaybackExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    private val pendingTokenBuffer = java.lang.StringBuilder()
    private lateinit var flushPendingTokens: java.lang.Runnable

    private var ttsOutput: AudioTrack? = null
    private var currentState = UIState.INIT
    private var llmPromptText = java.lang.StringBuilder()

    private lateinit var loadModelLayout: ConstraintLayout
    private lateinit var chatLayout: ConstraintLayout
    private lateinit var loadModelText: TextView
    private lateinit var loadModelProgress: ProgressBar
    private lateinit var intelligenceToggleGroup: com.google.android.material.button.MaterialButtonToggleGroup
    private lateinit var wakeWordToggleGroup: com.google.android.material.button.MaterialButtonToggleGroup
    private lateinit var startSessionButton: Button

    private lateinit var chatRecyclerView: RecyclerView
    private lateinit var messageAdapter: MessageAdapter
    private lateinit var statusText: TextView
    private lateinit var voiceStateView: VoiceStateView
    private lateinit var loadNewModelButton: ImageButton
    private lateinit var clearTextButton: ImageButton

    private val modelSelection = registerForActivityResult(ActivityResultContracts.OpenDocument()) { selectedUri ->
        updateUIState(UIState.LOADING_MODEL)
        if (selectedUri == null) {
            updateUIState(UIState.INIT)
            mainHandler.post { loadModelText.text = "No file selected" }
            return@registerForActivityResult
        }

        engineExecutor.submit {
            val llmModelFile = extractModelFile(selectedUri)
            if (llmModelFile == null || !llmModelFile.exists()) {
                updateUIState(UIState.INIT)
                mainHandler.post { loadModelText.text = "Unable to access selected file" }
                return@submit
            }
            initEnginesOnDevice(llmModelFile)
        }
    }

    @SuppressLint("DefaultLocale", "SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.main_layout)

        loadModelLayout = findViewById(R.id.loadModelLayout)
        chatLayout = findViewById(R.id.chatLayout)
        loadModelText = findViewById(R.id.loadModelText)
        loadModelProgress = findViewById(R.id.loadModelProgress)

        intelligenceToggleGroup = findViewById(R.id.intelligenceToggleGroup)
        intelligenceToggleGroup.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (isChecked) {
                selectedMode = if (checkedId == R.id.cloudButton) Mode.CLOUD else Mode.ON_DEVICE
                updateStartButtonText()
            }
        }

        wakeWordToggleGroup = findViewById(R.id.wakeWordToggleGroup)
        wakeWordToggleGroup.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (isChecked) {
                selectedWakeWordProvider = if (checkedId == R.id.picovoiceButton) WakeWordProvider.PICOVOICE else WakeWordProvider.OPENWAKEWORD
            }
        }

        startSessionButton = findViewById(R.id.startSessionButton)
        startSessionButton.setOnClickListener {
            if (selectedMode == Mode.ON_DEVICE) {
                modelSelection.launch(arrayOf("application/octet-stream"))
            } else {
                updateUIState(UIState.LOADING_MODEL)
                engineExecutor.submit { initEnginesCloud() }
            }
        }

        updateStartButtonText()
        updateUIState(UIState.INIT)

        messageAdapter = MessageAdapter()
        flushPendingTokens = java.lang.Runnable {
            val chunk: String
            synchronized(pendingTokenBuffer) {
                chunk = pendingTokenBuffer.toString()
                pendingTokenBuffer.setLength(0)
            }
            if (chunk.isNotEmpty()) {
                messageAdapter.appendToLast(chunk)
                scrollToBottom()
            }
        }
        chatRecyclerView = findViewById(R.id.chatRecyclerView)
        val layoutManager = LinearLayoutManager(this)
        layoutManager.stackFromEnd = true
        chatRecyclerView.layoutManager = layoutManager
        chatRecyclerView.adapter = messageAdapter
        (chatRecyclerView.itemAnimator as? SimpleItemAnimator)?.supportsChangeAnimations = false

        statusText = findViewById(R.id.statusText)
        voiceStateView = findViewById(R.id.voiceStateView)

        loadNewModelButton = findViewById(R.id.loadNewModelButton)
        loadNewModelButton.setOnClickListener {
            resetEngines()
            updateUIState(UIState.INIT)
            mainHandler.post { messageAdapter.clear() }
        }

        clearTextButton = findViewById(R.id.clearButton)
        clearTextButton.setOnClickListener {
            engineExecutor.submit {
                if (selectedMode == Mode.ON_DEVICE) {
                    try {
                        dialog = picollm?.dialogBuilder?.setSystem(SYSTEM_PROMPT)?.build()
                    } catch (e: PicoLLMException) {
                        updateUIState(UIState.WAKE_WORD)
                        mainHandler.post {
                            messageAdapter.addMessage(Message(Message.Role.ASSISTANT, e.toString()))
                        }
                    }
                } else {
                    conversationHistory.clear()
                    conversationHistory.add(ChatRequestSystemMessage(SYSTEM_PROMPT))
                }
            }
            mainHandler.post {
                messageAdapter.clear()
                clearTextButton.isEnabled = false
                clearTextButton.setImageDrawable(
                    ResourcesCompat.getDrawable(resources, R.drawable.clear_button_disabled, null)
                )
            }
        }
    }

    private fun updateStartButtonText() {
        startSessionButton.text = if (selectedMode == Mode.CLOUD) "Initialize Keva" else "Select .pllm & Start"
    }

    private fun initEnginesCloud() {
        resetEngines()
        
        mainHandler.post { loadModelText.text = "Loading Wake Word Engine..." }
        wakeWordEngine = when (selectedWakeWordProvider) {
            WakeWordProvider.PICOVOICE -> PicovoiceWakeWordEngine(ACCESS_KEY)
            WakeWordProvider.OPENWAKEWORD -> OpenWakeWordEngine()
        }
        wakeWordEngine?.start(applicationContext) { keywordIndex ->
            if (keywordIndex == 0) {
                interrupt()
                llmPromptText = java.lang.StringBuilder()
                updateUIState(UIState.STT)
            }
        }

        mainHandler.post { loadModelText.text = "Loading Cheetah..." }
        try {
            cheetah = Cheetah.Builder()
                .setAccessKey(ACCESS_KEY)
                .setModelPath(STT_MODEL_FILE)
                .setEnableAutomaticPunctuation(true)
                .build(applicationContext)
        } catch (e: CheetahException) {
            onEngineInitError(e.message)
            return
        }

        if (chatClient == null) {
            mainHandler.post { loadModelText.text = "Initializing cloud LLM client..." }
            chatClient = ChatCompletionsClientBuilder()
                .credential(bearerCredential)
                .endpoint(OPENAI_ENDPOINT)
                .buildClient()
        }
        conversationHistory.clear()
        conversationHistory.add(ChatRequestSystemMessage(SYSTEM_PROMPT))

        mainHandler.post { loadModelText.text = "Loading Orca..." }
        try {
            orca = Orca.Builder()
                .setAccessKey(ACCESS_KEY)
                .setModelPath(TTS_MODEL_FILE)
                .build(applicationContext)
        } catch (e: OrcaException) {
            onEngineInitError(e.message)
            return
        }

        updateUIState(UIState.WAKE_WORD)
        voiceProcessor.addFrameListener { runWakeWordSTT(it) }
        voiceProcessor.addErrorListener { error -> onEngineProcessError(error.message) }
        startWakeWordListening()
    }

    private fun initEnginesOnDevice(modelFile: File) {
        resetEngines()
        
        mainHandler.post { loadModelText.text = "Loading Wake Word Engine..." }
        wakeWordEngine = when (selectedWakeWordProvider) {
            WakeWordProvider.PICOVOICE -> PicovoiceWakeWordEngine(ACCESS_KEY)
            WakeWordProvider.OPENWAKEWORD -> OpenWakeWordEngine()
        }
        wakeWordEngine?.start(applicationContext) { keywordIndex ->
            if (keywordIndex == 0) {
                interrupt()
                llmPromptText = java.lang.StringBuilder()
                updateUIState(UIState.STT)
            }
        }

        mainHandler.post { loadModelText.text = "Loading Cheetah..." }
        try {
            cheetah = Cheetah.Builder()
                .setAccessKey(ACCESS_KEY)
                .setModelPath(STT_MODEL_FILE)
                .setEnableAutomaticPunctuation(true)
                .build(applicationContext)
        } catch (e: CheetahException) {
            onEngineInitError(e.message)
            return
        }

        mainHandler.post { loadModelText.text = "Loading picoLLM..." }
        try {
            picollm = PicoLLM.Builder()
                .setAccessKey(ACCESS_KEY)
                .setModelPath(modelFile.absolutePath)
                .build()
            dialog = picollm?.dialogBuilder?.setSystem(SYSTEM_PROMPT)?.build()
        } catch (e: PicoLLMException) {
            onEngineInitError(e.message)
            return
        }

        mainHandler.post { loadModelText.text = "Loading Orca..." }
        try {
            orca = Orca.Builder()
                .setAccessKey(ACCESS_KEY)
                .setModelPath(TTS_MODEL_FILE)
                .build(applicationContext)
        } catch (e: OrcaException) {
            onEngineInitError(e.message)
            return
        }

        updateUIState(UIState.WAKE_WORD)
        voiceProcessor.addFrameListener { runWakeWordSTT(it) }
        voiceProcessor.addErrorListener { error -> onEngineProcessError(error.message) }
        startWakeWordListening()
    }

    private fun runWakeWordSTT(frame: ShortArray) {
        if (currentState == UIState.WAKE_WORD || currentState == UIState.LLM_TTS) {
            wakeWordEngine?.process(frame)
        } else if (currentState == UIState.STT) {
            try {
                val result = cheetah?.process(frame) ?: return
                llmPromptText.append(result.transcript)
                mainHandler.post {
                    messageAdapter.appendToLast(result.transcript)
                    scrollToBottom()
                }

                if (result.isEndpoint) {
                    val finalResult = cheetah?.flush() ?: return
                    llmPromptText.append(finalResult.transcript)
                    mainHandler.post {
                        messageAdapter.appendToLast(finalResult.transcript)
                        scrollToBottom()
                    }
                    runLLM(llmPromptText.toString())
                }
            } catch (e: CheetahException) {
                onEngineProcessError(e.message)
            }
        }
    }

    private fun runLLM(prompt: String) {
        if (prompt.isEmpty()) return

        val isQueueingTokens = AtomicBoolean(false)
        val tokensReadyLatch = CountDownLatch(1)
        val tokenQueue = ConcurrentLinkedQueue<String>()

        val isQueueingPcm = AtomicBoolean(false)
        val pcmReadyLatch = CountDownLatch(1)
        val pcmQueue = ConcurrentLinkedQueue<ShortArray>()

        updateUIState(UIState.LLM_TTS)
        mainHandler.post { messageAdapter.addMessage(Message(Message.Role.ASSISTANT, "")) }

        if (selectedMode == Mode.ON_DEVICE) {
            engineExecutor.submit {
                val picoLLMProfiler = TPSProfiler()
                try {
                    isQueueingTokens.set(true)
                    wasInterrupted.set(false)

                    dialog?.addHumanRequest(prompt)
                    finalCompletion = picollm?.generate(
                        dialog?.prompt ?: "",
                        PicoLLMGenerateParams.Builder()
                            .setStreamCallback { token ->
                                picoLLMProfiler.tock()
                                if (!token.isNullOrEmpty()) {
                                    val containsStopPhrase = STOP_PHRASES.any { token.contains(it) }
                                    if (!containsStopPhrase && currentState == UIState.LLM_TTS) {
                                        tokenQueue.add(token)
                                        tokensReadyLatch.countDown()
                                        synchronized(pendingTokenBuffer) {
                                            pendingTokenBuffer.append(token)
                                        }
                                        mainHandler.removeCallbacks(flushPendingTokens)
                                        mainHandler.postDelayed(flushPendingTokens, 16)
                                    }
                                }
                            }
                            .setCompletionTokenLimit(COMPLETION_TOKEN_LIMIT)
                            .setStopPhrases(STOP_PHRASES)
                            .build()
                    )
                    dialog?.addLLMResponse(finalCompletion?.completion ?: "")
                    if (finalCompletion?.endpoint == PicoLLMCompletion.Endpoint.INTERRUPTED) {
                        wasInterrupted.set(true)
                    }
                    Log.i("PICOVOICE", String.format("TPS: %.2f", picoLLMProfiler.tps()))

                    isQueueingTokens.set(false)
                    mainHandler.removeCallbacks(flushPendingTokens)
                    mainHandler.post(flushPendingTokens)

                    mainHandler.post {
                        clearTextButton.isEnabled = true
                        clearTextButton.setImageDrawable(
                            ResourcesCompat.getDrawable(resources, R.drawable.clear_button, null)
                        )
                    }
                } catch (e: PicoLLMException) {
                    onEngineProcessError(e.message)
                }
            }
        } else {
            engineExecutor.submit {
                try {
                    isQueueingTokens.set(true)
                    interruptLLM.set(false)
                    wasInterrupted.set(false)

                    conversationHistory.add(ChatRequestUserMessage(prompt))
                    val opts = ChatCompletionsOptions(ArrayList(conversationHistory))
                    opts.model = OPENAI_MODEL

                    val stream = chatClient?.completeStream(opts)
                    val fullResponse = StringBuilder()
                    
                    if (stream != null) {
                        for (update in stream) {
                            if (interruptLLM.get()) {
                                wasInterrupted.set(true)
                                break
                            }
                            if (CoreUtils.isNullOrEmpty(update.choices)) continue
                            
                            val delta = update.choices[0].delta ?: continue
                            val token = delta.content
                            if (!token.isNullOrEmpty() && currentState == UIState.LLM_TTS) {
                                tokenQueue.add(token)
                                tokensReadyLatch.countDown()
                                fullResponse.append(token)

                                synchronized(pendingTokenBuffer) {
                                    pendingTokenBuffer.append(token)
                                }
                                mainHandler.removeCallbacks(flushPendingTokens)
                                mainHandler.postDelayed(flushPendingTokens, 16)
                            }
                        }
                    }

                    conversationHistory.add(ChatRequestAssistantMessage(fullResponse.toString()))
                    isQueueingTokens.set(false)

                    mainHandler.removeCallbacks(flushPendingTokens)
                    mainHandler.post(flushPendingTokens)

                    mainHandler.post {
                        clearTextButton.isEnabled = true
                        clearTextButton.setImageDrawable(
                            ResourcesCompat.getDrawable(resources, R.drawable.clear_button, null)
                        )
                    }
                } catch (e: Exception) {
                    onEngineProcessError(e.message)
                }
            }
        }

        ttsSynthesizeExecutor.submit {
            val orc = orca ?: return@submit
            val orcaStream: Orca.OrcaStream
            try {
                orcaStream = orc.streamOpen(OrcaSynthesizeParams.Builder().build())
            } catch (e: OrcaException) {
                onEngineProcessError(e.message)
                return@submit
            }

            val orcaProfiler = RTFProfiler(orc.sampleRate)
            var warmupPcm: ShortArray? = if (TTS_WARMUP_SECONDS > 0) ShortArray(0) else null

            try {
                tokensReadyLatch.await()
            } catch (e: InterruptedException) {
                onEngineProcessError(e.message)
                return@submit
            }

            isQueueingPcm.set(true)
            while (isQueueingTokens.get() || !tokenQueue.isEmpty()) {
                val token = tokenQueue.poll()
                if (!token.isNullOrEmpty()) {
                    try {
                        orcaProfiler.tick()
                        val pcm = orcaStream.synthesize(token)
                        orcaProfiler.tock(pcm)

                        if (pcm != null && pcm.isNotEmpty()) {
                            if (warmupPcm != null) {
                                val offset = warmupPcm.size
                                warmupPcm = warmupPcm.copyOf(offset + pcm.size)
                                System.arraycopy(pcm, 0, warmupPcm, offset, pcm.size)
                                if (warmupPcm.size > TTS_WARMUP_SECONDS * orc.sampleRate) {
                                    pcmQueue.add(warmupPcm)
                                    pcmReadyLatch.countDown()
                                    warmupPcm = null
                                }
                            } else {
                                pcmQueue.add(pcm)
                                pcmReadyLatch.countDown()
                            }
                        }
                    } catch (e: OrcaException) {
                        onEngineProcessError(e.message)
                        return@submit
                    }
                } else {
                    try { Thread.sleep(1) } catch (_: InterruptedException) {}
                }
            }

            try {
                orcaProfiler.tick()
                val flushedPcm = orcaStream.flush()
                orcaProfiler.tock(flushedPcm)

                if (flushedPcm != null && flushedPcm.isNotEmpty()) {
                    if (warmupPcm != null) {
                        val offset = warmupPcm.size
                        warmupPcm = warmupPcm.copyOf(offset + flushedPcm.size)
                        System.arraycopy(flushedPcm, 0, warmupPcm, offset, flushedPcm.size)
                        pcmQueue.add(warmupPcm)
                        pcmReadyLatch.countDown()
                    } else {
                        pcmQueue.add(flushedPcm)
                        pcmReadyLatch.countDown()
                    }
                }
                Log.i("PICOVOICE", String.format("RTF: %.2f", orcaProfiler.rtf()))
            } catch (e: OrcaException) {
                onEngineProcessError(e.message)
            }

            isQueueingPcm.set(false)
            orcaStream.close()
        }

        ttsPlaybackExecutor.submit {
            val orc = orca ?: return@submit
            try {
                val audioAttributes = AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                    .build()

                val audioFormat = AudioFormat.Builder()
                    .setSampleRate(orc.sampleRate)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()

                ttsOutput = AudioTrack(
                    audioAttributes,
                    audioFormat,
                    AudioTrack.getMinBufferSize(orc.sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT) * 4,
                    AudioTrack.MODE_STREAM,
                    0
                )
                ttsOutput?.play()
            } catch (e: Exception) {
                onEngineProcessError(e.message)
                return@submit
            }

            try {
                pcmReadyLatch.await()
            } catch (e: InterruptedException) {
                onEngineProcessError(e.message)
                return@submit
            }

            while (isQueueingPcm.get() || !pcmQueue.isEmpty()) {
                val pcm = pcmQueue.poll()
                if (pcm != null && pcm.isNotEmpty() && ttsOutput?.playState == AudioTrack.PLAYSTATE_PLAYING) {
                    ttsOutput?.write(pcm, 0, pcm.size)
                }
            }

            if (ttsOutput?.playState == AudioTrack.PLAYSTATE_PLAYING) {
                ttsOutput?.flush()
                ttsOutput?.stop()
            }
            ttsOutput?.release()
            ttsOutput = null

            if (wasInterrupted.get()) {
                llmPromptText = java.lang.StringBuilder()
                updateUIState(UIState.STT)
            } else {
                updateUIState(UIState.WAKE_WORD)
            }
        }
    }

    private fun interrupt() {
        if (selectedMode == Mode.ON_DEVICE) {
            try {
                picollm?.interrupt()
            } catch (e: PicoLLMException) {
                onEngineProcessError(e.message)
            }
        } else {
            interruptLLM.set(true)
        }
        if (ttsOutput?.playState == AudioTrack.PLAYSTATE_PLAYING) {
            ttsOutput?.stop()
        }
    }

    private fun resetEngines() {
        try {
            voiceProcessor.stop()
        } catch (e: VoiceProcessorException) {
            Log.e("PICOVOICE", "Error stopping voice processor", e)
        }
        voiceProcessor.clearFrameListeners()
        voiceProcessor.clearErrorListeners()

        wakeWordEngine?.stop()
        wakeWordEngine = null

        cheetah?.delete()
        cheetah = null

        picollm?.delete()
        picollm = null

        orca?.delete()
        orca = null
        
        conversationHistory.clear()
    }

    private fun extractModelFile(uri: Uri): File? {
        val modelFile = File(applicationContext.filesDir, "model.pllm")
        try {
            contentResolver.openInputStream(uri)?.use { inputStream ->
                FileOutputStream(modelFile).use { outputStream ->
                    val buffer = ByteArray(8192)
                    var numBytesRead: Int
                    while (inputStream.read(buffer).also { numBytesRead = it } != -1) {
                        outputStream.write(buffer, 0, numBytesRead)
                    }
                }
            }
        } catch (e: IOException) {
            return null
        }
        return modelFile
    }

    private fun onEngineInitError(message: String?) {
        updateUIState(UIState.INIT)
        mainHandler.post { loadModelText.text = message }
    }

    private fun onEngineProcessError(message: String?) {
        updateUIState(UIState.WAKE_WORD)
        mainHandler.post { messageAdapter.addMessage(Message(Message.Role.ASSISTANT, message ?: "")) }
    }

    private fun scrollToBottom() {
        val count = messageAdapter.itemCount
        if (count > 0) {
            chatRecyclerView.scrollToPosition(count - 1)
        }
    }

    private fun startWakeWordListening() {
        if (voiceProcessor.hasRecordAudioPermission(this)) {
            try {
                cheetah?.let { voiceProcessor.start(it.frameLength, it.sampleRate) }
            } catch (e: VoiceProcessorException) {
                onEngineProcessError(e.message)
            }
        } else {
            requestRecordPermission()
        }
    }

    private fun requestRecordPermission() {
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), 0)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults.isEmpty() || grantResults[0] == PackageManager.PERMISSION_DENIED) {
            onEngineProcessError("Recording permission not granted")
        } else {
            startWakeWordListening()
        }
    }

    private fun updateUIState(state: UIState) {
        mainHandler.post {
            when (state) {
                UIState.INIT -> {
                    loadModelLayout.visibility = View.VISIBLE
                    chatLayout.visibility = View.INVISIBLE
                    startSessionButton.isEnabled = true
                    startSessionButton.alpha = 1.0f
                    intelligenceToggleGroup.isEnabled = true
                    wakeWordToggleGroup.isEnabled = true
                    loadModelProgress.visibility = View.INVISIBLE
                    loadModelText.text = resources.getString(R.string.intro_text)
                }
                UIState.LOADING_MODEL -> {
                    loadModelLayout.visibility = View.VISIBLE
                    chatLayout.visibility = View.INVISIBLE
                    startSessionButton.isEnabled = false
                    startSessionButton.alpha = 0.5f
                    intelligenceToggleGroup.isEnabled = false
                    wakeWordToggleGroup.isEnabled = false
                    loadModelProgress.visibility = View.VISIBLE
                    loadModelText.text = "Loading Keva..."
                }
                UIState.WAKE_WORD -> {
                    loadModelLayout.visibility = View.INVISIBLE
                    chatLayout.visibility = View.VISIBLE
                    loadNewModelButton.setImageDrawable(ResourcesCompat.getDrawable(resources, R.drawable.arrow_back_button, null))
                    loadNewModelButton.isEnabled = true
                    voiceStateView.setState(VoiceStateView.State.WAKE_WORD)
                    statusText.text = if (selectedWakeWordProvider == WakeWordProvider.PICOVOICE) "Say 'Picovoice'!" else "Say the wake word!"
                    if (messageAdapter.itemCount > 0) {
                        clearTextButton.isEnabled = true
                        clearTextButton.setImageDrawable(ResourcesCompat.getDrawable(resources, R.drawable.clear_button, null))
                    } else {
                        clearTextButton.isEnabled = false
                        clearTextButton.setImageDrawable(ResourcesCompat.getDrawable(resources, R.drawable.clear_button_disabled, null))
                    }
                }
                UIState.STT -> {
                    loadModelLayout.visibility = View.INVISIBLE
                    chatLayout.visibility = View.VISIBLE
                    loadNewModelButton.setImageDrawable(ResourcesCompat.getDrawable(resources, R.drawable.arrow_back_button_disabled, null))
                    loadNewModelButton.isEnabled = false
                    voiceStateView.setState(VoiceStateView.State.STT)
                    statusText.text = "Listening..."
                    messageAdapter.addMessage(Message(Message.Role.USER, ""))
                    clearTextButton.isEnabled = true
                    clearTextButton.setImageDrawable(ResourcesCompat.getDrawable(resources, R.drawable.clear_button, null))
                }
                UIState.LLM_TTS -> {
                    loadModelLayout.visibility = View.INVISIBLE
                    chatLayout.visibility = View.VISIBLE
                    loadNewModelButton.setImageDrawable(ResourcesCompat.getDrawable(resources, R.drawable.arrow_back_button_disabled, null))
                    loadNewModelButton.isEnabled = false
                    voiceStateView.setState(VoiceStateView.State.LLM_TTS)
                    val wakeWord = if (selectedWakeWordProvider == WakeWordProvider.PICOVOICE) "'Picovoice'" else "wake word"
                    statusText.text = "Say $wakeWord to interrupt"
                    clearTextButton.isEnabled = false
                    clearTextButton.setImageDrawable(ResourcesCompat.getDrawable(resources, R.drawable.clear_button_disabled, null))
                }
            }
            currentState = state
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        engineExecutor.shutdownNow()
        ttsSynthesizeExecutor.shutdownNow()
        ttsPlaybackExecutor.shutdownNow()

        resetEngines()
    }

    companion object {
        private const val ACCESS_KEY = BuildConfig.PICOVOICE_ACCESS_KEY
        private const val GITHUB_TOKEN = BuildConfig.GITHUB_TOKEN
        private const val OPENAI_ENDPOINT = "https://models.github.ai/inference"
        private const val OPENAI_MODEL = "openai/gpt-4o"
        private const val STT_MODEL_FILE = "cheetah_params.pv"
        private const val TTS_MODEL_FILE = "orca_params_female.pv"
        private const val SYSTEM_PROMPT = "You are a voice assistant. Follow these rules strictly: " +
                "1. Keep all responses under 3 sentences unless the user explicitly asks for more detail. " +
                "2. Never use markdown, bullet points, numbered lists, or special characters  your response will be spoken aloud. " +
                "3. Speak naturally and conversationally, as if talking to a person. " +
                "4. If you don't know something, say so briefly. Never make up facts. " +
                "5. For simple questions, give a single direct sentence."
        private const val TTS_WARMUP_SECONDS = 1
        private const val COMPLETION_TOKEN_LIMIT = 128
        private val STOP_PHRASES = arrayOf(
            "</s>",
            "<end_of_turn>",
            "<|endoftext|>",
            "<|eot_id|>",
            "<|end|>", "<|user|>", "<|assistant|>"
        )
    }
}