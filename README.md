# Transformer 기반 HEAT 악성 URL 탐지 시스템
### (Transformer-based HEAT Malicious URL Detection System)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-ee4c2c)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![DeBERTa](https://img.shields.io/badge/Model-DeBERTa--v3--Large-orange)

## 📖 프로젝트 개요 (Abstract)
본 프로젝트는 기존의 블랙리스트 및 규칙 기반 방식의 한계를 극복하기 위해, **Transformer(DeBERTa)** 아키텍처를 활용하여 URL의 구조, 패턴, 의미적 문맥을 학습하는 지능형 악성 URL 탐지 시스템입니다. 자동화 도구로 생성된 변종 URL이나 교묘한 우회 패턴을 높은 수준의 적응성으로 판별하며, 실시간 웹 서비스 연동을 통해 실제 보안 환경에서 활용 가능한 솔루션을 제공합니다.

---

## 🛠 시스템 아키텍처 (System Architecture)
본 시스템은 데이터 처리부터 모델 학습, 그리고 실시간 웹 서비스 제공까지 단일 파이프라인으로 구성됩니다.

### 1. 데이터셋 구성 및 전처리 (Data Pipeline)
* **데이터 규모**: Kaggle, Alexa Top Sites, KISA 등에서 수집한 약 **178만 건**의 대규모 데이터셋 활용 
* **품질 확보**: 깨진 문자열 및 중복 항목 제거, 특수문자 정규화 및 소문자 변환 수행 
* **최적화**: 데이터 불균형 해소를 위해 정상/악성 데이터를 동일 수량으로 조정하고 랜덤 셔플 적용 

### 2. HEAT 악성 URL 탐지 모델 (DeBERTa Model)
* **모델 구조**: **DeBERTa-v3-Large** 아키텍처 기반 설계 [cite: 15]
* **학습 기법**: $EarlyStopping$, $Learning$ $Rate$ $Scheduler$, $Dropout$ 적용으로 과적합 방지 및 학습 안정화 
* **경량화**: **FP16 Mixed Precision** 기법을 적용하여 GPU 자원 효율성을 높이고 추론 속도를 대폭 개선

### 3. 실시간 탐지 웹 서비스 (Web Service)
* **프레임워크**: **FastAPI** 기반 백엔드 및 웹 사용자 인터페이스(UI) 구축 
* **기능**: 사용자가 URL 입력 시 실시간으로 위험도를 분석하여 예측 결과 및 위험도 제공 

---

## 📊 학습 결과 및 성능 (Performance)
v1~v3 단계의 고도화 과정을 통해 대규모 데이터셋에서 매우 안정적인 판별 능력을 확보했습니다.

* **$Accuracy$ (정확도)**: 약 **97%** 
* **$Precision$ (정밀도)**: 약 **96%** 
* **$Recall$ (재현율)**: 약 **95%** 
* **$F1$ $Score$**: 약 **96%** 
* **$AUC$**: **0.996** (정상과 악성 URL 사이의 명확한 구분 능력 증명) 

---
