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
    private val wakeWordModelFile: String = "alexa.onnx",
    private val threshold: Float = 0.5f,
    private val samplesPerChunk: Int = 1280
) : WakeWordEngine {

    private var env: OrtEnvironment? = null
    private var melSession: OrtSession? = null
    private var wwSession: OrtSession? = null

    private var wakeWordCallback: ((Int) -> Unit)? = null

    // Audio buffering
    private val audioBuffer = FloatArray(samplesPerChunk)
    private var audioBufferIdx = 0

    private val engineExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    // Feature buffering for the Wake Word Model
    // Commonly the wakeword model expects [1, 76] or [1, number_of_frames, 32]
    // We will dynamically determine this based on the ONNX model input shape
    private var featureBuffer = FloatArray(0)
    private var featureFramesRequired = 0
    private var featureSize = 0

    override fun start(context: Context, onWakeWordDetected: (keywordIndex: Int) -> Unit) {
        this.wakeWordCallback = onWakeWordDetected
        try {
            env = OrtEnvironment.getEnvironment()

            val options = SessionOptions()
            options.setIntraOpNumThreads(1) // Limit threads to reduce CPU contention

            // Load Mel-spectrogram Session
            val melModelBytes = context.assets.open(melspectrogramModelFile).readBytes()
            melSession = env?.createSession(melModelBytes, options)

            // Load Wake Word Session
            val wwModelBytes = context.assets.open(wakeWordModelFile).readBytes()
            wwSession = env?.createSession(wwModelBytes, options)

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
                    featureFramesRequired = 1
                    featureSize = shape[1].toInt()
                }
            } else {
                // Fallback for openWakeWord v0.1 defaults
                featureFramesRequired = 16
                featureSize = 76
            }

            featureBuffer = FloatArray(featureFramesRequired * featureSize)

        } catch (e: Exception) {
            Log.e("OpenWakeWordEngine", "Failed to initialize ONNX Runtime or load models", e)
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
            val audioShape = longArrayOf(1, samplesPerChunk.toLong())
            val audioBufferData = FloatBuffer.wrap(audioChunk)
            val melInputTensor = OnnxTensor.createTensor(env, audioBufferData, audioShape)

            val melResult = melSession?.run(mapOf(melInputName to melInputTensor))
            melInputTensor.close()

            // Output of mel model is usually [batch, features] or [1, 1, features]
            val melFeaturesObj = melResult?.get(0)?.value
            val melFeaturesFlat = when (melFeaturesObj) {
                is Array<*> -> flattenFloatArray(melFeaturesObj)
                is FloatArray -> melFeaturesObj
                else -> { melResult?.close(); return }
            }
            melResult.close()

            if (melFeaturesFlat.size != featureSize && featureSize > 0) {
                 // For openWakeword, the melspectrogram generates [1, 76] usually.
                 // We will just copy what we got.
            }

            // 2. Shift feature buffer left and add new features
            val fSize = melFeaturesFlat.size
            if (featureBuffer.size >= fSize) {
                System.arraycopy(featureBuffer, fSize, featureBuffer, 0, featureBuffer.size - fSize)
                System.arraycopy(melFeaturesFlat, 0, featureBuffer, featureBuffer.size - fSize, fSize)
            } else {
                // Buffer is smaller than features, just take what fits or re-alloc
               featureBuffer = melFeaturesFlat.clone()
            }

            // 3. Run Wake Word model
            val wwInputName = wwSession?.inputNames?.first() ?: return
            
            // Reconstruct shape expected by wake word model
            val wwInputInfo = wwSession?.inputInfo?.values?.first()
            val wwShape = (wwInputInfo?.info as? ai.onnxruntime.TensorInfo)?.shape 
                ?: longArrayOf(1, featureFramesRequired.toLong(), featureSize.toLong())
                
            // Fix any unknown dimensions (usually batch size at index 0)
            if (wwShape.isNotEmpty() && wwShape[0] == -1L) wwShape[0] = 1L
                
            val wwBufferData = FloatBuffer.wrap(featureBuffer)
            val wwInputTensor = OnnxTensor.createTensor(env, wwBufferData, wwShape)

            val wwResult = wwSession?.run(mapOf(wwInputName to wwInputTensor))
            wwInputTensor.close()

            // 4. Check Probability
            val probObj = wwResult?.get(0)?.value
            val probs = when (probObj) {
                is Array<*> -> flattenFloatArray(probObj)
                is FloatArray -> probObj
                else -> FloatArray(0)
            }
            wwResult?.close()

            if (probs.isNotEmpty() && probs[0] > threshold) {
                Log.i("OpenWakeWordEngine", "Wake word detected! Prob: ${probs[0]}")
                wakeWordCallback?.invoke(0)
                // Optionally clear the buffer to prevent immediate re-triggering
                featureBuffer.fill(0f)
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
        wwSession?.close()
        wwSession = null
        env?.close()
        env = null
    }
}
