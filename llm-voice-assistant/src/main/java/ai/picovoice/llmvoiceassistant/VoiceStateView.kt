package ai.picovoice.llmvoiceassistant

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import android.view.animation.DecelerateInterpolator
import android.view.animation.LinearInterpolator
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * Animated indicator showing the assistant's current mode:
 *
 *   WAKE_WORD - two rings expand and fade from a center dot (passive listening)
 *   STT       - five vertical bars animate like an audio spectrum (active capture)
 *   LLM_TTS   - a 270-degree arc spins continuously (generating/speaking)
 */
class VoiceStateView : View {

    enum class State {
        WAKE_WORD, STT, LLM_TTS
    }

    private var currentState = State.WAKE_WORD

    // Animated scalars fed by ValueAnimators
    private var pulseProgress = 0f
    private var barPhase = 0f
    private var arcAngle = 0f

    private val ringPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val barPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val arcPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val arcRect = RectF()

    private var pulseAnimator: ValueAnimator? = null
    private var barAnimator: ValueAnimator? = null
    private var rotateAnimator: ValueAnimator? = null

    companion object {
        private const val PRIMARY_COLOR = -0xc88201 // 0xFF377DFF
        private const val RING_RINGS = 2
        private const val BAR_COUNT = 5
        private val BAR_PHASE_OFFSETS = floatArrayOf(0f, 0.55f, 1.1f, 1.65f, 2.2f)
    }

    constructor(context: Context?) : super(context) { init() }
    constructor(context: Context?, attrs: AttributeSet?) : super(context, attrs) { init() }
    constructor(context: Context?, attrs: AttributeSet?, defStyleAttr: Int) : super(context, attrs, defStyleAttr) { init() }

    private fun init() {
        ringPaint.style = Paint.Style.STROKE
        ringPaint.strokeWidth = dp(2.5f)
        ringPaint.color = PRIMARY_COLOR

        dotPaint.style = Paint.Style.FILL
        dotPaint.color = PRIMARY_COLOR

        barPaint.style = Paint.Style.FILL
        barPaint.color = PRIMARY_COLOR

        arcPaint.style = Paint.Style.STROKE
        arcPaint.strokeWidth = dp(5f)
        arcPaint.color = PRIMARY_COLOR
        arcPaint.strokeCap = Paint.Cap.ROUND

        startAnimatorForState(State.WAKE_WORD)
    }

    fun setState(state: State) {
        if (state == currentState) return
        stopAllAnimators()
        currentState = state
        startAnimatorForState(state)
        invalidate()
    }

    private fun stopAllAnimators() {
        pulseAnimator?.cancel(); pulseAnimator = null
        barAnimator?.cancel(); barAnimator = null
        rotateAnimator?.cancel(); rotateAnimator = null
    }

    private fun startAnimatorForState(state: State) {
        when (state) {
            State.WAKE_WORD -> {
                pulseAnimator = ValueAnimator.ofFloat(0f, 1f).apply {
                    duration = 2200
                    repeatCount = ValueAnimator.INFINITE
                    repeatMode = ValueAnimator.RESTART
                    interpolator = DecelerateInterpolator(1.5f)
                    addUpdateListener { a ->
                        pulseProgress = a.animatedValue as Float
                        invalidate()
                    }
                    start()
                }
            }
            State.STT -> {
                barAnimator = ValueAnimator.ofFloat(0f, (Math.PI * 2).toFloat()).apply {
                    duration = 900
                    repeatCount = ValueAnimator.INFINITE
                    repeatMode = ValueAnimator.RESTART
                    interpolator = LinearInterpolator()
                    addUpdateListener { a ->
                        barPhase = a.animatedValue as Float
                        invalidate()
                    }
                    start()
                }
            }
            State.LLM_TTS -> {
                rotateAnimator = ValueAnimator.ofFloat(0f, 360f).apply {
                    duration = 1100
                    repeatCount = ValueAnimator.INFINITE
                    repeatMode = ValueAnimator.RESTART
                    interpolator = LinearInterpolator()
                    addUpdateListener { a ->
                        arcAngle = a.animatedValue as Float
                        invalidate()
                    }
                    start()
                }
            }
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val cx = width / 2f
        val cy = height / 2f

        when (currentState) {
            State.WAKE_WORD -> drawPulseRings(canvas, cx, cy)
            State.STT -> drawEqualiserBars(canvas, cx, cy)
            State.LLM_TTS -> drawRotatingArc(canvas, cx, cy)
        }
    }

    private fun drawPulseRings(canvas: Canvas, cx: Float, cy: Float) {
        val maxR = min(cx, cy) * 0.85f
        val dotR = maxR * 0.18f
        val ringGap = 0.45f

        for (i in 0 until RING_RINGS) {
            val phase = (pulseProgress + i * ringGap) % 1f
            val alpha = (255 * (1f - phase) * (1f - phase)).toInt()
            val radius = dotR + phase * (maxR - dotR)
            ringPaint.alpha = alpha
            canvas.drawCircle(cx, cy, radius, ringPaint)
        }

        dotPaint.alpha = 255
        canvas.drawCircle(cx, cy, dotR, dotPaint)
    }

    private fun drawEqualiserBars(canvas: Canvas, cx: Float, cy: Float) {
        val maxBarH = min(cx, cy) * 1.1f
        val barW = maxBarH * 0.14f
        val spacing = barW * 0.65f
        val totalW = BAR_COUNT * barW + (BAR_COUNT - 1) * spacing
        val startX = cx - totalW / 2f
        val cornerR = barW / 2f

        barPaint.alpha = 255
        for (i in 0 until BAR_COUNT) {
            val sinVal = sin((barPhase + BAR_PHASE_OFFSETS[i]).toDouble())
            var height = (sinVal * 0.5 + 0.5).toFloat() * maxBarH
            height = max(height, maxBarH * 0.12f)

            val left = startX + i * (barW + spacing)
            val right = left + barW
            val top = cy - height / 2f
            val bottom = cy + height / 2f

            canvas.drawRoundRect(left, top, right, bottom, cornerR, cornerR, barPaint)
        }
    }

    private fun drawRotatingArc(canvas: Canvas, cx: Float, cy: Float) {
        val radius = min(cx, cy) * 0.62f
        arcRect.set(cx - radius, cy - radius, cx + radius, cy + radius)
        arcPaint.alpha = 255
        canvas.drawArc(arcRect, arcAngle, 270f, false, arcPaint)
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        stopAllAnimators()
    }

    private fun dp(dp: Float): Float {
        return dp * resources.displayMetrics.density
    }
}
