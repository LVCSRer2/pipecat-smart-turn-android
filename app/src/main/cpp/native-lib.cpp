#include <jni.h>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <android/log.h>

#define TAG "SmartTurnNative"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

// FFT Constants
const int N_FFT = 512;
const int WINDOW_SIZE = 400;
const int N_MELS = 80;
const int HOP_LENGTH = 160;
const int N_FREQ_BINS = N_FFT / 2 + 1;

class NativeFFT {
private:
    int n;
    std::vector<float> cos_table;
    std::vector<float> sin_table;
    std::vector<int> bit_rev_table;

public:
    NativeFFT(int _n) : n(_n) {
        cos_table.resize(n / 2);
        sin_table.resize(n / 2);
        bit_rev_table.resize(n);

        for (int i = 0; i < n / 2; i++) {
            cos_table[i] = cosf(2.0f * M_PI * i / n);
            sin_table[i] = sinf(2.0f * M_PI * i / n);
        }

        int j = 0;
        for (int i = 0; i < n; i++) {
            bit_rev_table[i] = j;
            int k = n >> 1;
            while (k <= j && k > 0) {
                j -= k;
                k = k >> 1;
            }
            j += k;
        }
    }

    void forward(float* re, float* im) {
        for (int i = 0; i < n; i++) {
            int j = bit_rev_table[i];
            if (i < j) {
                std::swap(re[i], re[j]);
                std::swap(im[i], im[j]);
            }
        }

        for (int size = 2; size <= n; size <<= 1) {
            int half_size = size / 2;
            int step = n / size;
            for (int i = 0; i < n; i += size) {
                for (int k = 0; k < half_size; k++) {
                    float w_re = cos_table[k * step];
                    float w_im = -sin_table[k * step];
                    float tr = re[i + k + half_size] * w_re - im[i + k + half_size] * w_im;
                    float ti = re[i + k + half_size] * w_im + im[i + k + half_size] * w_re;
                    re[i + k + half_size] = re[i + k] - tr;
                    im[i + k + half_size] = im[i + k] - ti;
                    re[i + k] += tr;
                    im[i + k] += ti;
                }
            }
        }
    }
};

static NativeFFT* global_fft = nullptr;
static std::vector<float> hann_window;

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_smartturn_MelSpectrogram_extractMelNative(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray audio_data,
    jfloatArray mel_filters_data,
    jint n_frames) {

    if (!global_fft) {
        global_fft = new NativeFFT(N_FFT);
        hann_window.resize(WINDOW_SIZE);
        for (int i = 0; i < WINDOW_SIZE; i++) {
            hann_window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / WINDOW_SIZE));
        }
    }

    jfloat* audio = env->GetFloatArrayElements(audio_data, nullptr);
    jfloat* mel_filters = env->GetFloatArrayElements(mel_filters_data, nullptr);
    jsize audio_size = env->GetArrayLength(audio_data);

    // Padding input to 8 seconds (128,000 samples)
    std::vector<float> input(128000, 0.0f);
    if (audio_size < 128000) {
        std::copy(audio, audio + audio_size, input.begin() + (128000 - audio_size));
    } else {
        std::copy(audio + (audio_size - 128000), audio + audio_size, input.begin());
    }

    std::vector<float> result(N_MELS * n_frames, 0.0f);
    float re[N_FFT];
    float im[N_FFT];
    float power[N_FREQ_BINS];

    for (int frame = 0; frame < n_frames; frame++) {
        int offset = frame * HOP_LENGTH;
        if (offset + WINDOW_SIZE > 128000) break;

        for (int i = 0; i < N_FFT; i++) {
            if (i < WINDOW_SIZE) {
                re[i] = input[offset + i] * hann_window[i];
            } else {
                re[i] = 0.0f;
            }
            im[i] = 0.0f;
        }

        global_fft->forward(re, im);

        // Power Spectrum for 201 bins
        for (int i = 0; i < 201; i++) {
            power[i] = re[i] * re[i] + im[i] * im[i];
        }

        // Mel Transformation (Sparse but using full matrix for simplicity in C++ first, still fast)
        for (int mel_idx = 0; mel_idx < N_MELS; mel_idx++) {
            float mel_val = 0.0f;
            for (int bin_idx = 0; bin_idx < 201; bin_idx++) {
                // mel_filters is (201, 80)
                mel_val += power[bin_idx] * mel_filters[bin_idx * N_MELS + mel_idx];
            }
            result[mel_idx * n_frames + frame] = log10f(std::max(mel_val, 1e-10f));
        }
    }

    // Normalization
    float max_log_mel = -1e30f;
    for (float v : result) if (v > max_log_mel) max_log_mel = v;
    float floor_val = max_log_mel - 8.0f;

    for (int i = 0; i < result.size(); i++) {
        float val = std::max(result[i], floor_val);
        result[i] = (val + 4.0f) / 4.0f;
    }

    jfloatArray output = env->NewFloatArray(result.size());
    env->SetFloatArrayRegion(output, 0, result.size(), result.data());

    env->ReleaseFloatArrayElements(audio_data, audio, JNI_ABORT);
    env->ReleaseFloatArrayElements(mel_filters_data, mel_filters, JNI_ABORT);

    return output;
}
