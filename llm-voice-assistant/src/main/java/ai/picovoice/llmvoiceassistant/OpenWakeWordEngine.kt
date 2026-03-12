package ai.picovoice.llmvoiceassistant

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import android.content.Context
import android.util.Log
import java.nio.FloatBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class OpenWakeWordEngine(
    private val melspectrogramModelFile: String = "melspectrogram.onnx",
    private val embeddingModelFile: String = "embedding_model.onnx",
    private val wakeWordModelFile: String = "alexa.onnx",
    private val threshold: Float = 0.3f,
    private val samplesPerChunk: Int = 1280
) : WakeWordEngine {

    private var env: OrtEnvironment? = null
    private var melSession: OrtSession? = null
    private var embSession: OrtSession? = null
    private var wwSession: OrtSession? = null

    private var wakeWordCallback: ((Int) -> Unit)? = null

   // Audio buffering
    private val audioBuffer = FloatArray(samplesPerChunk)
    private var audioBufferIdx = 0

    private var lastDetectionTime: Long = 0
    private val cooldownMs: Long = 1000

    private val engineExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    // === FIX: Mel frame buffer (76 frames × 32 features) ===
    private val MEL_FRAMES_REQUIRED = 76
    private val MEL_FEATURE_SIZE = 32
    private val melFrameBuffer = FloatArray(MEL_FRAMES_REQUIRED * MEL_FEATURE_SIZE)

    // Feature buffer for Wake Word model
    private var featureBuffer = FloatArray(0)
    private var featureFramesRequired = 16
    private var featureSize = 96

    private var processedChunkCount = 0

    override fun start(
        context: Context,
        onError: (String) -> Unit,
        onWakeWordDetected: (keywordIndex: Int) -> Unit
    ) {
        Log.i("OpenWakeWordEngine", "Starting OpenWakeWordEngine...")
        this.wakeWordCallback = onWakeWordDetected
        try {
            env = OrtEnvironment.getEnvironment()

            val options = SessionOptions()
            options.setIntraOpNumThreads(1) // Limit threads to reduce CPU contention

            // Load Mel-spectrogram Session
            val melModelBytes = try {
                context.assets.open(melspectrogramModelFile).readBytes()
            } catch (e: Exception) {
                onError("Failed to load mel model asset: ${e.message}")
                return
            }
            melSession = env?.createSession(melModelBytes, options)

            // Load Embedding Session
            val embModelBytes = try {
                context.assets.open(embeddingModelFile).readBytes()
            } catch (e: Exception) {
                onError("Failed to load embedding model asset: ${e.message}")
                return
            }
            embSession = env?.createSession(embModelBytes, options)

            // Load Wake Word Session
            val wwModelBytes = try {
                context.assets.open(wakeWordModelFile).readBytes()
            } catch (e: Exception) {
                onError("Failed to load wake word model asset: ${e.message}")
                return
            }
            wwSession = env?.createSession(wwModelBytes, options)

            // Log shapes for debugging
            melSession?.inputInfo?.forEach { (name, info) ->
                val shape = (info.info as? ai.onnxruntime.TensorInfo)?.shape
                Log.i("OpenWakeWordEngine", "Mel input: $name, shape: ${shape?.contentToString()}")
            }
            embSession?.inputInfo?.forEach { (name, info) ->
                val shape = (info.info as? ai.onnxruntime.TensorInfo)?.shape
                Log.i("OpenWakeWordEngine", "Emb input: $name, shape: ${shape?.contentToString()}")
            }
            wwSession?.inputInfo?.forEach { (name, info) ->
                val shape = (info.info as? ai.onnxruntime.TensorInfo)?.shape
                Log.i("OpenWakeWordEngine", "WW input: $name, shape: ${shape?.contentToString()}")
            }

            // Dynamically set up feature buffers based on the Wakeword model's expected input
            val wwInputInfo = wwSession?.inputInfo?.values?.first()
            val shape = (wwInputInfo?.info as? ai.onnxruntime.TensorInfo)?.shape
            
            // Expected shape is often [batch_size, frames, features] e.g., [1, 16, 76] or [1, 76]
            if (shape != null && shape.size >= 2) {
                // If it's a 3D tensor like [1, frames, features]
                if (shape.size == 3) {
                    featureFramesRequired = shape[1].toInt()
                    featureSize = shape[2].toInt()
                } else if (shape.size == 2) {
                    featureFramesRequired = shape[0].toInt()  // e.g. 16
                    featureSize = shape[1].toInt()
                }
            } else {
                // Fallback for openWakeWord v0.1 defaults
                featureFramesRequired = 16
                featureSize = 76
            }

            featureBuffer = FloatArray(featureFramesRequired * featureSize)
            Log.i("OpenWakeWordEngine", "OpenWakeWordEngine initialized successfully")

        } catch (e: Exception) {
            Log.e("OpenWakeWordEngine", "Failed to initialize ONNX Runtime or load models", e)
            onError(e.message ?: "Failed to initialize OpenWakeWordEngine")
        }
    }


    override fun process(frame: ShortArray) {
        if (melSession == null || wwSession == null || env == null) return

        // Convert 16-bit PCM to normalizes Floats
        for (sample in frame) {
            audioBuffer[audioBufferIdx++] = sample / 32768.0f

            if (audioBufferIdx == samplesPerChunk) {
                val chunkToProcess = audioBuffer.clone()
                engineExecutor.submit { processAudioChunk(chunkToProcess) }
                audioBufferIdx = 0
            }
        }
    }

    private fun processAudioChunk(audioChunk: FloatArray) {
        try {
            // 1. Run Mel-spectrogram model
            val melInputName = melSession?.inputNames?.first() ?: return
            val melShape = longArrayOf(1, audioChunk.size.toLong())

            val audioBufferData = FloatBuffer.wrap(audioChunk)
            val melInputTensor = OnnxTensor.createTensor(env, audioBufferData, melShape)
            val melResult = melSession?.run(mapOf(melInputName to melInputTensor))
            melInputTensor.close()

            val melFeaturesObj = melResult?.get(0)?.value
            val melFeaturesFlat = when (melFeaturesObj) {
                is Array<*> -> flattenFloatArray(melFeaturesObj)
                is FloatArray -> melFeaturesObj
                else -> { melResult?.close(); return }
            }
            melResult?.close()

            // melFeaturesFlat is [newFrames * 32], e.g. 5 frames = 160 elements
            val newFrameCount = melFeaturesFlat.size / MEL_FEATURE_SIZE
            if (newFrameCount == 0) return

            // === FIX: Shift mel frame buffer left, append new frames ===
            val newFeatureBytes = newFrameCount * MEL_FEATURE_SIZE
            System.arraycopy(melFrameBuffer, newFeatureBytes, melFrameBuffer, 0, melFrameBuffer.size - newFeatureBytes)
            System.arraycopy(melFeaturesFlat, 0, melFrameBuffer, melFrameBuffer.size - newFeatureBytes, newFeatureBytes)

            // 2. Run Embedding model with shape [1, 76, 32, 1]
            val embInputName = embSession?.inputNames?.first() ?: return
            val embShape = longArrayOf(1, MEL_FRAMES_REQUIRED.toLong(), MEL_FEATURE_SIZE.toLong(), 1L)

            val melBufferData = FloatBuffer.wrap(melFrameBuffer)
            val melInputTensorForEmb = OnnxTensor.createTensor(env, melBufferData, embShape)
            val embResult = embSession?.run(mapOf(embInputName to melInputTensorForEmb))
            melInputTensorForEmb.close()

            val embFeaturesObj = embResult?.get(0)?.value
            val embFeaturesFlat = when (embFeaturesObj) {
                is Array<*> -> flattenFloatArray(embFeaturesObj)
                is FloatArray -> embFeaturesObj
                else -> { embResult?.close(); return }
            }
            embResult?.close()

            // 3. Shift embedding feature buffer, append new embedding (96 features)
            val fSize = embFeaturesFlat.size
            if (featureBuffer.size >= fSize) {
                System.arraycopy(featureBuffer, fSize, featureBuffer, 0, featureBuffer.size - fSize)
                System.arraycopy(embFeaturesFlat, 0, featureBuffer, featureBuffer.size - fSize, fSize)
            } else {
                featureBuffer = embFeaturesFlat.clone()
            }

            // 4. Run Wake Word model with shape [1, 16, 96]
            val wwInputName = wwSession?.inputNames?.first() ?: return
            val wwShape = longArrayOf(1, featureFramesRequired.toLong(), featureSize.toLong())

            val wwBufferData = FloatBuffer.wrap(featureBuffer)
            val wwInputTensor = OnnxTensor.createTensor(env, wwBufferData, wwShape)
            val wwResult = wwSession?.run(mapOf(wwInputName to wwInputTensor))
            wwInputTensor.close()

            val probObj = wwResult?.get(0)?.value
            val probs = when (probObj) {
                is Array<*> -> flattenFloatArray(probObj)
                is FloatArray -> probObj
                else -> FloatArray(0)
            }
            wwResult?.close()

            if (probs.isNotEmpty() && probs[0] > threshold) {
                val currentTime = System.currentTimeMillis()
                if (currentTime - lastDetectionTime > cooldownMs) {
                    Log.i("OpenWakeWordEngine", "Wake word detected! Prob: ${probs[0]}")
                    lastDetectionTime = currentTime
                    wakeWordCallback?.invoke(0)
                    melFrameBuffer.fill(0f)
                    featureBuffer.fill(0f)
                }
            }

        } catch (e: Exception) {
            Log.e("OpenWakeWordEngine", "Error during processAudioChunk", e)
        }
    }
    
    // Helper to recursively flatten N-dimensional array into FloatArray
    private fun flattenFloatArray(array: Array<*>): FloatArray {
        val result = mutableListOf<Float>()
        for (item in array) {
            when (item) {
                is FloatArray -> result.addAll(item.toList())
                is Array<*> -> result.addAll(flattenFloatArray(item).toList())
                is Float -> result.add(item)
            }
        }
        return result.toFloatArray()
    }

    override fun stop() {
        Log.i("OpenWakeWordEngine", "Stopping OpenWakeWordEngine...")
        wakeWordCallback = null
        
        engineExecutor.shutdown()
        try {
            if (!engineExecutor.awaitTermination(500, TimeUnit.MILLISECONDS)) {
                engineExecutor.shutdownNow()
            }
        } catch (e: InterruptedException) {
            engineExecutor.shutdownNow()
        }

        melSession?.close()
        melSession = null
        embSession?.close()
        embSession = null
        wwSession?.close()
        wwSession = null
        env?.close()
        env = null
    }
}
