# Pipecat Smart Turn Android Demo

이 프로젝트는 [Pipecat Smart Turn v3.1](https://github.com/pipecat-ai/smart-turn) 모델을 안드로이드에서 로컬(On-Device)로 실행하여 실시간 발화 완료(End-of-Turn, EOT)를 감지하는 데모입니다.

## 주요 특징

- **실시간 음성 감지 (VAD)**: [Silero VAD](https://github.com/snakers4/silero-vad) ONNX 모델을 사용하여 마이크 입력에서 음성 구간을 실시간으로 감지합니다.
- **고성능 특징 추출 (Mel Spectrogram)**: Whisper 모델 규격에 맞는 Log-Mel Spectrogram 추출 로직을 **C++ NDK** 및 최적화된 **Kotlin**으로 구현했습니다.
- **스마트 턴 감지 (Smart Turn)**: 사용자가 말을 멈춘 후 1초(1000ms)의 침묵이 발생하면, 전체 발화 문맥을 분석하여 AI가 응답할 타이밍인지(EOT) 아니면 아직 생각 중인 상태인지(Continued) 판별합니다.
- **On-Device 추론**: 모든 연산이 기기 내에서 처리되므로 프라이버시가 보호되고 레이턴시가 낮습니다.

## 성능 비교 (800 프레임 분석 기준)

실시간성을 확보하기 위해 전처리 로직을 최적화하였으며, 결과는 다음과 같습니다.

| 방식 | 평균 소요 시간 | 특징 |
| :--- | :--- | :--- |
| **Kotlin (Optimized)** | ~70 ms | Sparse Matrix 연산 및 FFT 최적화 적용 |
| **Native (C++ NDK)** | **~25 ms** | whisper.cpp 스타일의 고성능 구현 |

## 아키텍처

1. **AudioRecord**: 16kHz Mono PCM 오디오 캡처.
2. **Silero VAD**: 512 샘플 단위로 음성 여부 판단.
3. **MelSpectrogram (Native)**: 발화 종료 시점부터 과거 8초간의 데이터를 Mel 특징으로 변환.
4. **Smart Turn ONNX**: 특징 맵을 입력받아 턴 종료 확률 계산.

## 설치 및 실행

1. 저장소를 클론합니다.
2. Android Studio에서 프로젝트를 엽니다.
3. `local.properties`에 Android SDK 경로가 올바른지 확인합니다.
4. 빌드 및 기기 설치: `./gradlew installDebug`

## 기술 스택

- **언어**: Kotlin, C++ (JNI/NDK)
- **추론 엔진**: ONNX Runtime Android (v1.19.0)
- **빌드 시스템**: Gradle 8.2, CMake
- **머신러닝 모델**:
  - Silero VAD (ONNX)
  - Smart Turn v3.1 (ONNX, Whisper Tiny 기반)

## 참고 자료

- [Pipecat Smart Turn GitHub](https://github.com/pipecat-ai/smart-turn)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) (FFT/Mel 추출 로직 참고)
