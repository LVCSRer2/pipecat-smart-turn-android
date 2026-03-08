package com.example.smartturn

import android.Manifest
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Color
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.preference.PreferenceManager
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.*
import java.nio.FloatBuffer

class MainActivity : AppCompatActivity() {

    private companion object {
        const val REQUEST_RECORD_AUDIO_PERMISSION = 200
        const val SAMPLE_RATE = 16000
        const val CHUNK_SIZE = 512
        const val TAG = "SmartTurnDemo"
    }

    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var recordingJob: Job? = null

    private lateinit var statusLabel: TextView
    private lateinit var probabilityText: TextView
    private lateinit var recordButton: Button
    private var defaultTextColor: Int = Color.BLACK

    private lateinit var sileroVad: SileroVAD
    private lateinit var melSpectrogram: MelSpectrogram
    private lateinit var ortEnv: OrtEnvironment
    private lateinit var smartTurnSession: OrtSession
    private lateinit var prefs: SharedPreferences

    private val speechSegment = mutableListOf<Float>()
    private var trailingSilenceSamples = 0
    private var isSpeechActive = false

    // Dynamic Settings
    private var stopMs = 1000
    private var vadThreshold = 0.5f
    private var eotThreshold = 0.5f
    private var maxWindowSec = 8
    private var inferenceMode = "native"
    private var displayTimeMs = 2000

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusLabel = findViewById(R.id.statusLabel)
        probabilityText = findViewById(R.id.probabilityText)
        recordButton = findViewById(R.id.recordButton)
        defaultTextColor = statusLabel.currentTextColor

        prefs = PreferenceManager.getDefaultSharedPreferences(this)
        loadSettings()

        recordButton.setOnClickListener {
            if (isRecording) {
                stopRecording()
            } else {
                startRecordingWithPermission()
            }
        }

        initModels()
    }

    private fun loadSettings() {
        stopMs = prefs.getInt("trailing_silence", 1000)
        vadThreshold = prefs.getInt("vad_sensitivity", 50) / 100f
        eotThreshold = prefs.getInt("eot_threshold", 50) / 100f
        maxWindowSec = prefs.getString("max_window", "8")?.toInt() ?: 8
        inferenceMode = prefs.getString("inference_mode", "native") ?: "native"
        displayTimeMs = prefs.getInt("display_time", 2000)
        Log.d(TAG, "Settings loaded: stopMs=$stopMs, vad=$vadThreshold, eot=$eotThreshold, mode=$inferenceMode")
    }

    override fun onResume() {
        super.onResume()
        loadSettings()
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.main_menu, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        if (item.itemId == R.id.action_settings) {
            startActivity(Intent(this, SettingsActivity::class.java))
            return true
        }
        return super.onOptionsItemSelected(item)
    }

    private fun initModels() {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            sileroVad = SileroVAD(this)
            melSpectrogram = MelSpectrogram(this)
            val modelBytes = assets.open("smart-turn-v3.1-cpu.onnx").readBytes()
            smartTurnSession = ortEnv.createSession(modelBytes)
            updateStatus("Models Ready", defaultTextColor)
        } catch (e: Exception) {
            Log.e(TAG, "Error: ${e.message}")
            updateStatus("Error: Models missing", Color.RED)
        }
    }

    private fun updateStatus(text: String, color: Int) {
        runOnUiThread {
            statusLabel.text = "Status: $text"
            statusLabel.setTextColor(color)
        }
    }

    private fun startRecordingWithPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO_PERMISSION)
        } else {
            startRecording()
        }
    }

    private fun startRecording() {
        val bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) return
        
        audioRecord = AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufferSize)
        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) return

        audioRecord?.startRecording()
        isRecording = true
        runOnUiThread { recordButton.text = "Stop Recording" }
        updateStatus("Listening", defaultTextColor)

        recordingJob = CoroutineScope(Dispatchers.IO).launch {
            val readBuffer = ShortArray(CHUNK_SIZE)
            while (isActive && isRecording) {
                val readCount = audioRecord?.read(readBuffer, 0, CHUNK_SIZE) ?: 0
                if (readCount > 0) processAudioChunk(readBuffer, readCount)
            }
        }
    }

    private fun stopRecording() {
        isRecording = false
        recordingJob?.cancel()
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        runOnUiThread { recordButton.text = "Start Recording" }
        updateStatus("Stopped", defaultTextColor)
        isSpeechActive = false
        speechSegment.clear()
        trailingSilenceSamples = 0
    }

    private suspend fun processAudioChunk(shorts: ShortArray, count: Int) {
        val chunk = FloatArray(count)
        for (i in 0 until count) chunk[i] = shorts[i] / 32768.0f
        val prob = sileroVad.predict(chunk)
        
        if (prob > vadThreshold) {
            if (!isSpeechActive) {
                isSpeechActive = true
                updateStatus("Speech Detected", defaultTextColor)
            }
            trailingSilenceSamples = 0
        } else if (isSpeechActive) {
            trailingSilenceSamples += count
        }

        if (isSpeechActive) {
            for (f in chunk) speechSegment.add(f)
            val maxSamples = maxWindowSec * SAMPLE_RATE
            if (speechSegment.size > maxSamples) {
                repeat(speechSegment.size - maxSamples) { speechSegment.removeAt(0) }
            }

            if (trailingSilenceSamples >= (stopMs * SAMPLE_RATE / 1000)) {
                runInference()
                isSpeechActive = false
                speechSegment.clear()
                trailingSilenceSamples = 0
            }
        }
    }

    private suspend fun runInference() {
        if (speechSegment.isEmpty()) {
            updateStatus("No Speech", defaultTextColor)
            delay(1500)
            if (isRecording) updateStatus("Listening", defaultTextColor)
            return
        }
        val audio = speechSegment.toFloatArray()
        updateStatus("Analyzing...", defaultTextColor)

        val nFrames = maxWindowSec * 100
        var melData: FloatArray? = null
        var timeMel: Long = 0
        var timeKotlin: Long = -1

        when (inferenceMode) {
            "native" -> {
                val (res, time) = melSpectrogram.extractNative(audio, nFrames)
                melData = res
                timeMel = time
            }
            "kotlin" -> {
                val (res, time) = melSpectrogram.extractKotlin(audio, nFrames)
                melData = res
                timeMel = time
            }
            "both" -> {
                val (resK, tK) = melSpectrogram.extractKotlin(audio, nFrames)
                val (resN, tN) = melSpectrogram.extractNative(audio, nFrames)
                melData = resN
                timeMel = tN
                timeKotlin = tK
            }
        }

        var inferenceProb = 0.0f
        val startInf = System.currentTimeMillis()
        try {
            val inputTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(melData!!), longArrayOf(1, 80, nFrames.toLong()))
            val results = smartTurnSession.run(mapOf("input_features" to inputTensor))
            results.use {
                val output = results[0].value as Array<FloatArray>
                inferenceProb = output[0][0]
            }
        } catch (e: Exception) {
            Log.e(TAG, "Inference error: ${e.message}")
        }
        val timeInf = System.currentTimeMillis() - startInf

        withContext(Dispatchers.Main) {
            val resultText = StringBuilder()
            resultText.append("EOT Prob: %.4f\n".format(inferenceProb))
            if (inferenceMode == "both") {
                resultText.append("K: %d ms | N: %d ms\n".format(timeKotlin, timeMel))
            } else {
                resultText.append("Mel Extraction: %d ms\n".format(timeMel))
            }
            resultText.append("EOT Inference: %d ms".format(timeInf))
            probabilityText.text = resultText.toString()

            if (inferenceProb > eotThreshold) {
                updateStatus("Turn Complete!", Color.RED)
            } else {
                updateStatus("Continued", Color.BLUE)
            }
            
            delay(displayTimeMs.toLong())
            if (isRecording) updateStatus("Listening", defaultTextColor)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopRecording()
        sileroVad.close()
        if (::smartTurnSession.isInitialized) {
            smartTurnSession.close()
            ortEnv.close()
        }
    }
}
