package com.example.smartturn

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer

class SileroVAD(context: Context) {
    private val ortEnv = OrtEnvironment.getEnvironment()
    private val ortSession: OrtSession
    private val sr = longArrayOf(16000)
    
    private var state = FloatArray(2 * 1 * 128) { 0f }
    private var contextBuffer = FloatArray(64) { 0f }

    init {
        val modelBytes = context.assets.open("silero_vad.onnx").readBytes()
        ortSession = ortEnv.createSession(modelBytes)
    }

    /**
     * Predict speech probability for a 512-sample chunk (16kHz).
     */
    fun predict(chunk: FloatArray): Float {
        val inputData = FloatArray(64 + 512)
        System.arraycopy(contextBuffer, 0, inputData, 0, 64)
        System.arraycopy(chunk, 0, inputData, 64, 512)

        val inputTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(inputData), longArrayOf(1, 576))
        val stateTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(state), longArrayOf(2, 1, 128))
        val srTensor = OnnxTensor.createTensor(ortEnv, sr)

        val inputs = mapOf(
            "input" to inputTensor,
            "state" to stateTensor,
            "sr" to srTensor
        )

        val results = ortSession.run(inputs)
        results.use {
            val output = results[0].value as Array<FloatArray>
            val newState = results[1].value as Array<Array<FloatArray>>
            
            var pos = 0
            for (i in 0 until 2) {
                for (j in 0 until 1) {
                    for (k in 0 until 128) {
                        state[pos++] = newState[i][j][k]
                    }
                }
            }

            System.arraycopy(inputData, inputData.size - 64, contextBuffer, 0, 64)
            return output[0][0]
        }
    }

    fun close() {
        ortSession.close()
        ortEnv.close()
    }
}
