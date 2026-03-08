package com.example.smartturn

import android.content.Context
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

class MelSpectrogram(context: Context) {
    private val nMels = 80
    private val originalBins = 201
    private val melFilters = FloatArray(nMels * originalBins)

    init {
        System.loadLibrary("smartturn-native")
        
        val inputStream = context.assets.open("mel_filters.bin")
        val buffer = ByteArray(melFilters.size * 4)
        var bytesRead = 0
        while (bytesRead < buffer.size) {
            val read = inputStream.read(buffer, bytesRead, buffer.size - bytesRead)
            if (read == -1) break
            bytesRead += read
        }
        ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(melFilters)
        inputStream.close()
        
        Log.d("SmartTurnDemo", "Native MelSpectrogram initialized")
    }

    private external fun extractMelNative(audio: FloatArray, filters: FloatArray, nFrames: Int): FloatArray

    /**
     * Extracts Log-Mel Spectrogram using high-performance C++ implementation.
     */
    fun extractNative(audio: FloatArray, nFrames: Int = 800): Pair<FloatArray, Long> {
        val startTime = System.currentTimeMillis()
        val result = extractMelNative(audio, melFilters, nFrames)
        val duration = System.currentTimeMillis() - startTime
        return Pair(result, duration)
    }
}
