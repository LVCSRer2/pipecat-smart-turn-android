package com.example.smartturn

import android.content.Context
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

class MelSpectrogram(context: Context) {
    private val nMels = 80
    private val nFft = 512
    private val nFreqBins = nFft / 2 + 1
    private val hopLength = 160
    private val windowSize = 400
    private val originalBins = 201
    private val melFilters = FloatArray(nMels * originalBins)

    // Sparse Filter Storage for Kotlin
    private val filterIndices: IntArray
    private val filterWeights: FloatArray
    private val filterOffsets: IntArray
    private val filterLengths: IntArray

    private val fft = FFT(nFft)
    private val window = FloatArray(windowSize)

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

        // Prepare Sparse Filters for Kotlin
        val indicesList = mutableListOf<Int>()
        val weightsList = mutableListOf<Float>()
        filterOffsets = IntArray(nMels)
        filterLengths = IntArray(nMels)

        var currentOffset = 0
        for (melIdx in 0 until nMels) {
            filterOffsets[melIdx] = currentOffset
            var count = 0
            for (binIdx in 0 until originalBins) {
                val weight = melFilters[binIdx * nMels + melIdx]
                if (weight > 0.0001f) {
                    indicesList.add(binIdx)
                    weightsList.add(weight)
                    count++
                    currentOffset++
                }
            }
            filterLengths[melIdx] = count
        }
        filterIndices = indicesList.toIntArray()
        filterWeights = weightsList.toFloatArray()

        for (i in 0 until windowSize) {
            window[i] = 0.5f * (1 - cos(2 * PI * i / windowSize)).toFloat()
        }
        
        Log.d("SmartTurnDemo", "Dual MelSpectrogram (Kotlin + Native) initialized")
    }

    private external fun extractMelNative(audio: FloatArray, filters: FloatArray, nFrames: Int): FloatArray

    /**
     * Native C++ Implementation
     */
    fun extractNative(audio: FloatArray, nFrames: Int = 800): Pair<FloatArray, Long> {
        val startTime = System.currentTimeMillis()
        val result = extractMelNative(audio, melFilters, nFrames)
        val duration = System.currentTimeMillis() - startTime
        return Pair(result, duration)
    }

    /**
     * Optimized Kotlin Implementation
     */
    fun extractKotlin(audio: FloatArray, nFrames: Int = 800): Pair<FloatArray, Long> {
        val startTime = System.currentTimeMillis()
        val result = FloatArray(nMels * nFrames)
        
        val input = FloatArray(128000)
        if (audio.size < 128000) {
            System.arraycopy(audio, 0, input, 128000 - audio.size, audio.size)
        } else {
            System.arraycopy(audio, audio.size - 128000, input, 0, 128000)
        }

        val re = FloatArray(nFft)
        val im = FloatArray(nFft)
        val power = FloatArray(nFreqBins)

        for (frame in 0 until nFrames) {
            val offset = frame * hopLength
            if (offset + windowSize > input.size) break

            for (i in 0 until nFft) {
                if (i < windowSize) {
                    re[i] = input[offset + i] * window[i]
                } else {
                    re[i] = 0f
                }
                im[i] = 0f
            }

            fft.forward(re, im)

            for (i in 0 until originalBins) {
                power[i] = re[i] * re[i] + im[i] * im[i]
            }

            for (melIdx in 0 until nMels) {
                var melValue = 0f
                val start = filterOffsets[melIdx]
                val end = start + filterLengths[melIdx]
                for (i in start until end) {
                    melValue += power[filterIndices[i]] * filterWeights[i]
                }
                result[melIdx * nFrames + frame] = log10(max(melValue, 1e-10f))
            }
        }

        var maxLogMel = -Float.MAX_VALUE
        for (i in result.indices) if (result[i] > maxLogMel) maxLogMel = result[i]
        val floor = maxLogMel - 8.0f
        for (i in result.indices) {
            result[i] = (max(result[i], floor) + 4.0f) / 4.0f
        }
        
        val duration = System.currentTimeMillis() - startTime
        return Pair(result, duration)
    }
}
