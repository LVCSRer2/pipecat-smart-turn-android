package com.example.smartturn

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
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
        const val STOP_MS = 1000
        const val MAX_DURATION_SAMPLES = 8 * 16000 // 8 seconds
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

    private val speechSegment = mutableListOf<Float>()
    private var trailingSilenceSamples = 0
    private var isSpeechActive = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusLabel = findViewById(R.id.statusLabel)
        probabilityText = findViewById(R.id.probabilityText)
        recordButton = findViewById(R.id.recordButton)
        defaultTextColor = statusLabel.currentTextColor

        recordButton.setOnClickListener {
            if (isRecording) {
                stopRecording()
            } else {
                startRecordingWithPermission()
            }
        }

        initModels()
    }

    private fun initModels() {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            sileroVad = SileroVAD(this)
            melSpectrogram = MelSpectrogram(this)
            
            val modelBytes = assets.open("smart-turn-v3.1-cpu.onnx").readBytes()
            smartTurnSession = ortEnv.createSession(modelBytes)
            
            Log.d(TAG, "Models initialized successfully")
            updateStatus("Models Ready", defaultTextColor)
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing models: ${e.message}")
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
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO_PERMISSION)
        } else {
            startRecording()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_RECORD_AUDIO_PERMISSION && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startRecording()
        }
    }

    private fun startRecording() {
        val bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            return
        }
        
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
                if (readCount > 0) {
                    processAudioChunk(readBuffer, readCount)
                }
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
        
        if (prob > 0.5f) {
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
            if (speechSegment.size > MAX_DURATION_SAMPLES) {
                repeat(speechSegment.size - MAX_DURATION_SAMPLES) { speechSegment.removeAt(0) }
            }

            if (trailingSilenceSamples >= (STOP_MS * SAMPLE_RATE / 1000)) {
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

        // 1. Native Feature Extraction
        val (melNative, timeMel) = melSpectrogram.extractNative(audio)

        // 2. Smart Turn ONNX Inference (EOT Determination)
        var inferenceProb = 0.0f
        val startInf = System.currentTimeMillis()
        try {
            val inputTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(melNative), longArrayOf(1, 80, 800))
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
            resultText.append("Mel Extraction: %d ms\n".format(timeMel))
            resultText.append("EOT Inference: %d ms".format(timeInf))
            probabilityText.text = resultText.toString()

            if (inferenceProb > 0.5) {
                updateStatus("Turn Complete!", Color.RED)
            } else {
                updateStatus("Continued", Color.BLUE)
            }
            
            // Return to Listening after 2 seconds
            delay(2000)
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
