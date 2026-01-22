# Transformer 기반 HEAT 악성 URL 탐지 시스템
### (Transformer-based HEAT Attack URL Detection System)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-ee4c2c)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![DeBERTa](https://img.shields.io/badge/Model-DeBERTa--v3--Large-orange)

## 📖 프로젝트 개요 (Abstract)
기존 블랙리스트 방식의 한계를 극복하기 위해, URL의 구조와 의미적 문맥을 학습하는 **Transformer(DeBERTa)** 기반의 지능형 악성 URL 탐지 시스템을 구현했습니다. 대규모 데이터셋 학습과 FP16 경량화를 통해 실시간 탐지가 가능한 웹 서비스 형태로 구축되었습니다.

## 🚀 주요 특징 (Key Features)
* [cite_start]**대규모 데이터 활용**: Kaggle, KISA 등 다양한 출처에서 수집한 약 178만 건의 URL 데이터셋 활용 [cite: 9, 10]
* [cite_start]**최첨단 모델 적용**: **DeBERTa-v3-Large** 아키텍처를 기반으로 변조된 URL 패턴 탐지 [cite: 15]
* [cite_start]**실시간 탐지 서비스**: FastAPI 백엔드와 웹 UI를 연동하여 URL 입력 시 즉각적인 위험도 제공 [cite: 26, 27]
* [cite_start]**최적화**: FP16 Mixed Precision 및 경량화를 통한 추론 속도 개선 [cite: 17, 26]

---

## 📊 모델 성능 (Performance)
[cite_start]v1~v3 단계의 고도화를 거쳐 최종적으로 매우 높은 판별 성능을 확보했습니다[cite: 18, 20].

* [cite_start]**$Accuracy$**: 약 97% [cite: 24]
* [cite_start]**$Precision$**: 약 96% [cite: 24]
* [cite_start]**$Recall$**: 약 95% [cite: 24]
* [cite_start]**$F1 Score$**: 약 96% [cite: 24]
* [cite_start]**$AUC$**: **0.996** [cite: 24]

---

## 🛠 시스템 아키텍처 (System Architecture)
1. [cite_start]**Data Pipeline**: 178만 건의 데이터 정제 및 정규화 [cite: 5, 9, 12]
2. [cite_start]**Model Training**: DeBERTa-v3-Large 기반 학습 (EarlyStopping, Scheduler 적용) 
3. [cite_start]**Web Integration**: FastAPI 백엔드 연동 및 실시간 분석 UI 구축 

---

## 👨‍💻 Contributors (참여자)
[cite_start]본 프로젝트는 **상명대학교 정보보안공학과** 캡스톤 디자인 팀 프로젝트(팀명: 원팀)로 진행되었습니다[cite: 1].

### 👩‍💻 Team Members (개발 팀원)
* **곽지현 (Me)**: [여기에 본인의 핵심 역할을 적어주세요. 예: DeBERTa 모델 설계 및 학습 주도 / FastAPI 백엔드 구축 등]
* [cite_start]**김예지 (팀장)**: 데이터 수집 파이프라인 구축 및 프로젝트 총괄 [cite: 40]
* [cite_start]**고가은**: 데이터 전처리 정교화 및 성능 평가 지표 분석 [cite: 1]

> [cite_start]**공동 작업**: 178만 건 데이터셋 확보, 모델 v1~v3 성능 고도화, 결과보고서 작성 [cite: 10, 18, 40]

### 🎓 Advisor (지도교수)
* [cite_start]**박진성 교수님** [cite: 1]
