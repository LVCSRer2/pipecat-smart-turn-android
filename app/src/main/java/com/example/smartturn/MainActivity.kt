package com.example.smartturn

import android.Manifest
import android.content.pm.PackageManager
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
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing models: ${e.message}")
            statusLabel.text = "Error: Models not found"
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
        recordButton.text = "Stop Recording"
        statusLabel.text = "Status: Listening..."

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
        recordButton.text = "Start Recording"
        statusLabel.text = "Status: Stopped"
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
                withContext(Dispatchers.Main) { statusLabel.text = "Status: Speech Detected" }
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
                runComparison()
                isSpeechActive = false
                speechSegment.clear()
                trailingSilenceSamples = 0
            }
        }
    }

    private suspend fun runComparison() {
        if (speechSegment.isEmpty()) return
        val audio = speechSegment.toFloatArray()

        withContext(Dispatchers.Main) { statusLabel.text = "Status: Comparing Kotlin vs Native..." }

        // 1. Kotlin Extraction
        val (melKotlin, timeKotlin) = melSpectrogram.extractKotlin(audio)
        
        // 2. Native Extraction
        val (melNative, timeNative) = melSpectrogram.extractNative(audio)

        // 3. ONNX Inference (using native result)
        var inferenceProb = 0.0f
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

        withContext(Dispatchers.Main) {
            val resultText = StringBuilder()
            resultText.append("EOT Prob: %.4f\n".format(inferenceProb))
            resultText.append("Kotlin: %d ms\n".format(timeKotlin))
            resultText.append("Native: %d ms\n".format(timeNative))
            resultText.append("Speedup: %.1fx".format(timeKotlin.toFloat() / timeNative))
            
            probabilityText.text = resultText.toString()
            statusLabel.text = if (inferenceProb > 0.5) "Status: Turn Complete!" else "Status: Continued"
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
