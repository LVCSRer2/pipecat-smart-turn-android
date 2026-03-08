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

    // Buffers
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
            statusLabel.text = "Error: Models not found in assets"
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

        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord initialization failed")
            return
        }

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
        for (i in 0 until count) {
            chunk[i] = shorts[i] / 32768.0f
        }

        val prob = sileroVad.predict(chunk)
        
        if (prob > 0.5f) {
            if (!isSpeechActive) {
                isSpeechActive = true
                withContext(Dispatchers.Main) {
                    statusLabel.text = "Status: Speech Detected"
                }
            }
            isSpeechActive = true
            trailingSilenceSamples = 0
        } else {
            if (isSpeechActive) {
                trailingSilenceSamples += count
            }
        }

        if (isSpeechActive) {
            for (f in chunk) speechSegment.add(f)
            
            // Limit buffer size to 8 seconds
            if (speechSegment.size > MAX_DURATION_SAMPLES) {
                val toRemove = speechSegment.size - MAX_DURATION_SAMPLES
                repeat(toRemove) { speechSegment.removeAt(0) }
            }

            // Check if silence lasted long enough to trigger Smart Turn
            if (trailingSilenceSamples >= (STOP_MS * SAMPLE_RATE / 1000)) {
                runSmartTurn()
                isSpeechActive = false
                speechSegment.clear()
                trailingSilenceSamples = 0
            }
        }
    }

    private suspend fun runSmartTurn() {
        if (speechSegment.isEmpty()) return

        withContext(Dispatchers.Main) {
            statusLabel.text = "Status: Analyzing turn..."
        }

        try {
            Log.d(TAG, "Starting Smart Turn analysis for ${speechSegment.size} samples")
            val audio = speechSegment.toFloatArray()
            Log.d(TAG, "Audio converted to FloatArray. Extracting Mel Spectrogram...")
            val mel = melSpectrogram.extract(audio)
            Log.d(TAG, "Mel Spectrogram extracted. Running ONNX inference...")
            
            // Smart Turn expects [1, 80, 800]
            val inputTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(mel), longArrayOf(1, 80, 800))
            
            val results = smartTurnSession.run(mapOf("input_features" to inputTensor))
            Log.d(TAG, "ONNX inference completed.")
            results.use {
                val output = results[0].value as Array<FloatArray>
                val probability = output[0][0]

                withContext(Dispatchers.Main) {
                    probabilityText.text = String.format("EOT Probability: %.4f", probability)
                    if (probability > 0.5) {
                        statusLabel.text = "Status: Turn Complete! (AI response)"
                    } else {
                        statusLabel.text = "Status: Continued (User still talking)"
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Smart Turn error: ${e.message}", e)
            withContext(Dispatchers.Main) {
                statusLabel.text = "Status: Error during analysis"
            }
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
