package com.example.smartturn

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

class FFT(private val n: Int) {
    private val cosTable = FloatArray(n / 2)
    private val sinTable = FloatArray(n / 2)
    private val bitRevTable = IntArray(n)

    init {
        for (i in 0 until n / 2) {
            cosTable[i] = cos(2 * PI * i / n).toFloat()
            sinTable[i] = sin(2 * PI * i / n).toFloat()
        }
        
        // Precompute bit reversal
        var j = 0
        for (i in 0 until n) {
            bitRevTable[i] = j
            var k = n shr 1
            while (k <= j && k > 0) {
                j -= k
                k = k shr 1
            }
            j += k
        }
    }

    fun forward(re: FloatArray, im: FloatArray) {
        // Bit-reversal swap using precomputed table
        for (i in 0 until n) {
            val j = bitRevTable[i]
            if (i < j) {
                val tempRe = re[i]
                val tempIm = im[i]
                re[i] = re[j]
                im[i] = im[j]
                re[j] = tempRe
                im[j] = tempIm
            }
        }

        // Cooley-Tukey iterative
        var size = 2
        while (size <= n) {
            val halfSize = size / 2
            val step = n / size
            for (i in 0 until n step size) {
                for (k in 0 until halfSize) {
                    val wRe = cosTable[k * step]
                    val wIm = -sinTable[k * step]
                    
                    val tr = re[i + k + halfSize] * wRe - im[i + k + halfSize] * wIm
                    val ti = re[i + k + halfSize] * wIm + im[i + k + halfSize] * wRe
                    
                    re[i + k + halfSize] = re[i + k] - tr
                    im[i + k + halfSize] = im[i + k] - ti
                    re[i + k] += tr
                    im[i + k] += ti
                }
            }
            size *= 2
        }
    }
}
