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

import android.animation.ValueAnimator;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;
import android.view.animation.DecelerateInterpolator;
import android.view.animation.LinearInterpolator;

/**
 * A stateful animated indicator that communicates the assistant's current mode
 * through three distinct visualizations:
 *
 *   WAKE_WORD  — two concentric rings slowly expand and fade from a center dot,
 *                suggesting passive, ambient listening (radar / sonar metaphor).
 *
 *   STT        — five vertical bars animate up and down like an audio spectrum,
 *                signalling active microphone capture.
 *
 *   LLM_TTS    — a 270° arc spins continuously, indicating that the assistant
 *                is generating and speaking a response.
 */
public class VoiceStateView extends View {

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    public enum State {
        WAKE_WORD,
        STT,
        LLM_TTS
    }

    public void setState(State state) {
        if (state == currentState) return;
        stopAllAnimators();
        currentState = state;
        startAnimatorForState(state);
        invalidate();
    }

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------

    private static final int PRIMARY_COLOR  = 0xFF377DFF;  // brand blue
    private static final int RING_RINGS     = 2;
    private static final int BAR_COUNT      = 5;
    // Phase offsets so adjacent bars move out of sync with each other
    private static final float[] BAR_PHASE_OFFSETS = {0f, 0.55f, 1.1f, 1.65f, 2.2f};

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------

    private State currentState = State.WAKE_WORD;

    // Animated scalars fed by ValueAnimators
    private float pulseProgress = 0f;   // 0..1 — overall pulse cycle position
    private float barPhase      = 0f;   // 0..2π — drives bar heights via sin()
    private float arcAngle      = 0f;   // 0..360 — start angle for rotating arc

    // -----------------------------------------------------------------------
    // Paint objects (created once, reused)
    // -----------------------------------------------------------------------

    private final Paint ringPaint  = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint dotPaint   = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint barPaint   = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint arcPaint   = new Paint(Paint.ANTI_ALIAS_FLAG);

    private final RectF arcRect = new RectF();

    // -----------------------------------------------------------------------
    // Animators
    // -----------------------------------------------------------------------

    private ValueAnimator pulseAnimator;
    private ValueAnimator barAnimator;
    private ValueAnimator rotateAnimator;

    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    public VoiceStateView(Context context) {
        super(context);
        init();
    }

    public VoiceStateView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public VoiceStateView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    // -----------------------------------------------------------------------
    // Initialisation
    // -----------------------------------------------------------------------

    private void init() {
        // Expanding, fading rings
        ringPaint.setStyle(Paint.Style.STROKE);
        ringPaint.setStrokeWidth(dp(2.5f));
        ringPaint.setColor(PRIMARY_COLOR);

        // Solid center dot
        dotPaint.setStyle(Paint.Style.FILL);
        dotPaint.setColor(PRIMARY_COLOR);

        // Equaliser bars
        barPaint.setStyle(Paint.Style.FILL);
        barPaint.setColor(PRIMARY_COLOR);

        // Rotating arc
        arcPaint.setStyle(Paint.Style.STROKE);
        arcPaint.setStrokeWidth(dp(5f));
        arcPaint.setColor(PRIMARY_COLOR);
        arcPaint.setStrokeCap(Paint.Cap.ROUND);

        // Start idle animation immediately
        startAnimatorForState(State.WAKE_WORD);
    }

    // -----------------------------------------------------------------------
    // Animator lifecycle
    // -----------------------------------------------------------------------

    private void stopAllAnimators() {
        if (pulseAnimator  != null) { pulseAnimator.cancel();  pulseAnimator  = null; }
        if (barAnimator    != null) { barAnimator.cancel();    barAnimator    = null; }
        if (rotateAnimator != null) { rotateAnimator.cancel(); rotateAnimator = null; }
    }

    private void startAnimatorForState(State state) {
        switch (state) {
            case WAKE_WORD:
                pulseAnimator = ValueAnimator.ofFloat(0f, 1f);
                pulseAnimator.setDuration(2200);
                pulseAnimator.setRepeatCount(ValueAnimator.INFINITE);
                pulseAnimator.setRepeatMode(ValueAnimator.RESTART);
                pulseAnimator.setInterpolator(new DecelerateInterpolator(1.5f));
                pulseAnimator.addUpdateListener(a -> {
                    pulseProgress = (float) a.getAnimatedValue();
                    invalidate();
                });
                pulseAnimator.start();
                break;

            case STT:
                barAnimator = ValueAnimator.ofFloat(0f, (float) (Math.PI * 2));
                barAnimator.setDuration(900);
                barAnimator.setRepeatCount(ValueAnimator.INFINITE);
                barAnimator.setRepeatMode(ValueAnimator.RESTART);
                barAnimator.setInterpolator(new LinearInterpolator());
                barAnimator.addUpdateListener(a -> {
                    barPhase = (float) a.getAnimatedValue();
                    invalidate();
                });
                barAnimator.start();
                break;

            case LLM_TTS:
                rotateAnimator = ValueAnimator.ofFloat(0f, 360f);
                rotateAnimator.setDuration(1100);
                rotateAnimator.setRepeatCount(ValueAnimator.INFINITE);
                rotateAnimator.setRepeatMode(ValueAnimator.RESTART);
                rotateAnimator.setInterpolator(new LinearInterpolator());
                rotateAnimator.addUpdateListener(a -> {
                    arcAngle = (float) a.getAnimatedValue();
                    invalidate();
                });
                rotateAnimator.start();
                break;
        }
    }

    // -----------------------------------------------------------------------
    // Drawing
    // -----------------------------------------------------------------------

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        final float cx = getWidth()  / 2f;
        final float cy = getHeight() / 2f;

        switch (currentState) {
            case WAKE_WORD: drawPulseRings(canvas, cx, cy); break;
            case STT:       drawEqualiserBars(canvas, cx, cy); break;
            case LLM_TTS:   drawRotatingArc(canvas, cx, cy); break;
        }
    }

    /**
     * WAKE_WORD — two staggered rings expand outward and fade, anchored to a
     * solid center dot. Calm, ambient; the dot signals "I'm here but quiet."
     */
    private void drawPulseRings(Canvas canvas, float cx, float cy) {
        final float maxR    = Math.min(cx, cy) * 0.85f;
        final float dotR    = maxR * 0.18f;
        final float ringGap = 0.45f; // controls spacing between the two rings

        for (int i = 0; i < RING_RINGS; i++) {
            float phase = (pulseProgress + i * ringGap) % 1f;
            int   alpha = (int) (255 * (1f - phase) * (1f - phase)); // quadratic fade
            float radius = dotR + phase * (maxR - dotR);
            ringPaint.setAlpha(alpha);
            canvas.drawCircle(cx, cy, radius, ringPaint);
        }

        // Solid center dot always at full opacity
        dotPaint.setAlpha(255);
        canvas.drawCircle(cx, cy, dotR, dotPaint);
    }

    /**
     * STT — five bars whose heights are driven by offset sin waves, producing
     * a rolling "audio spectrum" effect. Height is clamped to a minimum so
     * bars are always visible (avoiding a confusing flat line).
     */
    private void drawEqualiserBars(Canvas canvas, float cx, float cy) {
        final float maxBarH = Math.min(cx, cy) * 1.1f;
        final float barW    = maxBarH * 0.14f;
        final float spacing = barW * 0.65f;
        final float totalW  = BAR_COUNT * barW + (BAR_COUNT - 1) * spacing;
        final float startX  = cx - totalW / 2f;
        final float cornerR = barW / 2f;

        barPaint.setAlpha(255);
        for (int i = 0; i < BAR_COUNT; i++) {
            double sinVal = Math.sin(barPhase + BAR_PHASE_OFFSETS[i]);
            float  height = (float) (sinVal * 0.5 + 0.5) * maxBarH;
            height = Math.max(height, maxBarH * 0.12f); // minimum visible height

            float left   = startX + i * (barW + spacing);
            float right  = left + barW;
            float top    = cy - height / 2f;
            float bottom = cy + height / 2f;

            canvas.drawRoundRect(left, top, right, bottom, cornerR, cornerR, barPaint);
        }
    }

    /**
     * LLM_TTS — a 270° arc rotates continuously. The gap in the arc produces
     * comfortable motion cues without being distractingly busy.
     */
    private void drawRotatingArc(Canvas canvas, float cx, float cy) {
        final float radius = Math.min(cx, cy) * 0.62f;
        arcRect.set(cx - radius, cy - radius, cx + radius, cy + radius);
        arcPaint.setAlpha(255);
        canvas.drawArc(arcRect, arcAngle, 270f, false, arcPaint);
    }

    // -----------------------------------------------------------------------
    // Housekeeping
    // -----------------------------------------------------------------------

    @Override
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        stopAllAnimators();
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /** Convert dp to px using screen density. */
    private float dp(float dp) {
        return dp * getResources().getDisplayMetrics().density;
    }
}
