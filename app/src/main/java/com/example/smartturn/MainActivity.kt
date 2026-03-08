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
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.components.YAxis
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import kotlinx.coroutines.*
import java.nio.FloatBuffer

class MainActivity : AppCompatActivity() {

    private companion object {
        const val REQUEST_RECORD_AUDIO_PERMISSION = 200
        const val SAMPLE_RATE = 16000
        const val CHUNK_SIZE = 512
        const val MAX_CHART_POINTS = 300
        const val TAG = "SmartTurnDemo"
    }

    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var recordingJob: Job? = null

    private lateinit var statusLabel: TextView
    private lateinit var probabilityText: TextView
    private lateinit var recordButton: Button
    private lateinit var probabilityChart: LineChart
    
    private var defaultTextColor: Int = Color.BLACK

    private lateinit var sileroVad: SileroVAD
    private lateinit var melSpectrogram: MelSpectrogram
    private lateinit var ortEnv: OrtEnvironment
    private lateinit var smartTurnSession: OrtSession
    private lateinit var prefs: SharedPreferences

    private val speechSegment = mutableListOf<Float>()
    private var trailingSilenceSamples = 0
    private var isSpeechActive = false

    private lateinit var vadDataSet: LineDataSet
    private lateinit var eotDataSet: LineDataSet
    private lateinit var continuedDataSet: LineDataSet
    private var chartXIndex = 0f

    // Settings
    private var stopMs = 1000
    private var vadThreshold = 0.5f
    private var eotThreshold = 0.5f
    private var maxWindowSec = 8
    private var inferenceMode = "native"
    private var maintainContext = true
    private var displayTimeMs = 2000

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusLabel = findViewById(R.id.statusLabel)
        probabilityText = findViewById(R.id.probabilityText)
        recordButton = findViewById(R.id.recordButton)
        probabilityChart = findViewById(R.id.probabilityChart)
        defaultTextColor = statusLabel.currentTextColor

        initChart()
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

    private fun initChart() {
        probabilityChart.description.isEnabled = false
        probabilityChart.setTouchEnabled(false)
        probabilityChart.isDragEnabled = false
        probabilityChart.setScaleEnabled(false)
        probabilityChart.setPinchZoom(false)
        probabilityChart.setBackgroundColor(Color.WHITE)

        val data = LineData()
        probabilityChart.data = data

        vadDataSet = LineDataSet(mutableListOf(), "Speech (VAD)")
        vadDataSet.color = Color.GREEN
        vadDataSet.setDrawCircles(false)
        vadDataSet.setDrawValues(false)
        vadDataSet.lineWidth = 2f
        vadDataSet.axisDependency = YAxis.AxisDependency.LEFT
        data.addDataSet(vadDataSet)

        eotDataSet = LineDataSet(mutableListOf(), "Turn Complete")
        eotDataSet.color = Color.RED
        eotDataSet.setCircleColor(Color.RED)
        eotDataSet.circleRadius = 6f
        eotDataSet.setDrawCircleHole(false)
        eotDataSet.setDrawValues(false)
        eotDataSet.lineWidth = 0f
        eotDataSet.setDrawCircles(true)
        eotDataSet.axisDependency = YAxis.AxisDependency.LEFT
        data.addDataSet(eotDataSet)

        continuedDataSet = LineDataSet(mutableListOf(), "Continued")
        continuedDataSet.color = Color.BLUE
        continuedDataSet.setCircleColor(Color.BLUE)
        continuedDataSet.circleRadius = 6f
        continuedDataSet.setDrawCircleHole(false)
        continuedDataSet.setDrawValues(false)
        continuedDataSet.lineWidth = 0f
        continuedDataSet.setDrawCircles(true)
        continuedDataSet.axisDependency = YAxis.AxisDependency.LEFT
        data.addDataSet(continuedDataSet)

        val xl = probabilityChart.xAxis
        xl.setDrawGridLines(true)
        xl.isEnabled = true

        val leftAxis = probabilityChart.axisLeft
        leftAxis.axisMaximum = 1.1f
        leftAxis.axisMinimum = -0.1f
        leftAxis.setDrawGridLines(true)

        val rightAxis = probabilityChart.axisRight
        rightAxis.isEnabled = false
    }

    private fun loadSettings() {
        stopMs = prefs.getInt("trailing_silence", 1000)
        vadThreshold = prefs.getInt("vad_sensitivity", 50) / 100f
        eotThreshold = prefs.getInt("eot_threshold", 50) / 100f
        maxWindowSec = prefs.getString("max_window", "8")?.toInt() ?: 8
        inferenceMode = prefs.getString("inference_mode", "native") ?: "native"
        maintainContext = prefs.getBoolean("maintain_context", true)
        displayTimeMs = prefs.getInt("display_time", 2000)
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
            val modelBytes = assets.open("smart-turn-v3.2-cpu.onnx").readBytes()
            smartTurnSession = ortEnv.createSession(modelBytes)
            updateStatus("Models Ready", defaultTextColor)
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing: ${e.message}")
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

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_RECORD_AUDIO_PERMISSION && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startRecording()
        }
    }

    private fun startRecording() {
        try {
            val bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) return
            
            audioRecord = AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufferSize)
            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) return

            audioRecord?.startRecording()
            isRecording = true
            
            vadDataSet.clear()
            eotDataSet.clear()
            continuedDataSet.clear()
            chartXIndex = 0f
            probabilityChart.data.notifyDataChanged()
            probabilityChart.notifyDataSetChanged()
            probabilityChart.invalidate()

            runOnUiThread { recordButton.text = "Stop Recording" }
            updateStatus("Listening", defaultTextColor)

            recordingJob = CoroutineScope(Dispatchers.IO).launch {
                val readBuffer = ShortArray(CHUNK_SIZE)
                while (isActive && isRecording) {
                    val readCount = audioRecord?.read(readBuffer, 0, CHUNK_SIZE) ?: 0
                    if (readCount > 0) processAudioChunk(readBuffer, readCount)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Start recording error: ${e.message}")
        }
    }

    private fun stopRecording() {
        isRecording = false
        recordingJob?.cancel()
        try {
            audioRecord?.stop()
            audioRecord?.release()
        } catch (e: Exception) {
            Log.e(TAG, "Stop audio error: ${e.message}")
        }
        audioRecord = null
        runOnUiThread { recordButton.text = "Start Recording" }
        updateStatus("Stopped", defaultTextColor)
        isSpeechActive = false
        speechSegment.clear()
        trailingSilenceSamples = 0
    }

    private suspend fun processAudioChunk(shorts: ShortArray, count: Int) {
        if (!isRecording) return
        val chunk = FloatArray(count)
        for (i in 0 until count) chunk[i] = shorts[i] / 32768.0f
        val prob = sileroVad.predict(chunk)
        
        addChartEntry(prob)

        if (prob > vadThreshold) {
            if (!isSpeechActive) {
                isSpeechActive = true
                updateStatus("Speech Detected", defaultTextColor)
            }
            trailingSilenceSamples = 0
        } else if (isSpeechActive) {
            trailingSilenceSamples += count
        }

        // Always add to buffer if recording and speech has occurred
        if (isSpeechActive) {
            for (f in chunk) speechSegment.add(f)
            val maxSamples = maxWindowSec * SAMPLE_RATE
            if (speechSegment.size > maxSamples) {
                repeat(speechSegment.size - maxSamples) { speechSegment.removeAt(0) }
            }

            if (trailingSilenceSamples >= (stopMs * SAMPLE_RATE / 1000)) {
                runInference()
                // Silence threshold met, reset internal silence counter
                trailingSilenceSamples = 0
            }
        }
    }

    private fun addChartEntry(vadProb: Float) {
        if (!isRecording) return
        runOnUiThread {
            if (!isRecording) return@runOnUiThread
            val data = probabilityChart.data ?: return@runOnUiThread
            
            data.addEntry(Entry(chartXIndex, vadProb), 0)
            chartXIndex += 1f

            if (vadDataSet.entryCount > MAX_CHART_POINTS) {
                vadDataSet.removeFirst()
                val minX = vadDataSet.getEntryForIndex(0).x
                while (eotDataSet.entryCount > 0 && eotDataSet.getEntryForIndex(0).x < minX) {
                    eotDataSet.removeFirst()
                }
                while (continuedDataSet.entryCount > 0 && continuedDataSet.getEntryForIndex(0).x < minX) {
                    continuedDataSet.removeFirst()
                }
            }

            data.notifyDataChanged()
            probabilityChart.notifyDataSetChanged()
            probabilityChart.setVisibleXRangeMaximum(MAX_CHART_POINTS.toFloat())
            probabilityChart.moveViewToX(chartXIndex)
        }
    }

    private suspend fun runInference() {
        if (speechSegment.isEmpty() || !isRecording) return
        val audio = speechSegment.toFloatArray()
        val currentX = chartXIndex

        updateStatus("Analyzing...", defaultTextColor)

        val nFrames = maxWindowSec * 100
        var melData: FloatArray? = null
        var timeMel: Long = 0

        try {
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
                    val (resN, tN) = melSpectrogram.extractNative(audio, nFrames)
                    melData = resN
                    timeMel = tN
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Extraction error: ${e.message}")
            return
        }

        if (melData == null || !isRecording) return

        var inferenceProb = 0.0f
        val startInf = System.currentTimeMillis()
        try {
            val inputTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(melData), longArrayOf(1, 80, nFrames.toLong()))
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
            if (!isRecording) return@withContext
            
            val isEot = inferenceProb > eotThreshold
            if (isEot) {
                probabilityChart.data.addEntry(Entry(currentX, inferenceProb), 1)
                updateStatus("Turn Complete!", Color.RED)
            } else {
                probabilityChart.data.addEntry(Entry(currentX, inferenceProb), 2)
                updateStatus("Continued", Color.BLUE)
            }
            
            probabilityChart.data.notifyDataChanged()
            probabilityChart.notifyDataSetChanged()

            val resultText = "EOT Prob: %.4f\nMel: %d ms | EOT: %d ms".format(inferenceProb, timeMel, timeInf)
            probabilityText.text = resultText
            
            // Logic for maintaining context
            if (isEot || !maintainContext) {
                // If turn is finished OR maintain mode is OFF, clear buffer for next interaction
                speechSegment.clear()
                isSpeechActive = false
            } else {
                // Continued + maintain mode is ON: Keep speechSegment as is (Sliding window already handles size)
                Log.d(TAG, "Maintaining context for Continued utterance")
            }

            if (isRecording) {
                delay(100)
                updateStatus("Listening", defaultTextColor)
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
