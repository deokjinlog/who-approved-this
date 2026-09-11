# who-approved-this

공개 공공데이터로 문서 파싱 정확도를 재는 벤치마크 파이프라인.
이름 그대로 최종 목표는 "결재문서 던지면 누가 결재했는지 뽑는다"(T1).
회사 프로젝트(전자결재 AI — 결재문서 PDF에서 결재선·메타데이터 추출) 리허설용으로,
고객 데이터 없이 공개 데이터로 방법을 먼저 검증한다.
전체 계획은 `docs/plan.md`.

## 원칙

- **공통 구조는 collect → parse → evaluate 3단계.** 트랙은 각 단계의 어댑터만 추가하고,
  골격(base 클래스 인터페이스, 패키지 구조)은 바꾸지 않는다.
- **데이터는 리포 밖 `DATA_ROOT`**(`~/data/who-approved-this`, 환경변수 `WAT_DATA_ROOT`로 덮어쓰기).
  리포 안에 PDF·원문·중간 산출물을 커밋하지 않는다. 커밋하는 건 코드와 점수 JSON뿐.
- **결과표와 로그에 기안자 등 실명을 남기지 않는다.** 공공 결재문서에는 실명이 들어 있다.
  수집물은 로컬에만 두고, 결과 JSON·실패 케이스 샘플은 마스킹해서 기록한다.
  **마스킹 대상은 "사람 이름"뿐이다.** 직위·부서·기관명은 가리지 않는다(그게 지표다).
  카드번호·계좌·업체명은 마스킹 대상이 아니라 **애초에 결과에 담지 않는 것**으로 다룬다 —
  담을 이유가 있으면 그때 별도 규칙을 만든다.
  **예외(2026-09-11): 요약의 '업체' 칸** — 결재자가 확인할 정보라 요약 출력(API 응답,
  리포 밖 `summary_review/`)에는 담는다. 리포에 커밋하는 결과 JSON 에는 개수만 남긴다. 마스킹은 글자 사이가 벌어진 표기
  ("홍 길 동")도 잡아야 한다. 공문은 자간을 공백으로 벌려 찍는다.
- **첫 목표는 T4**: 관보 PDF 1장 → 200dpi 렌더링 → OCR → CER 숫자. 수집 자동화는 그 뒤에.
- **상용 API는 비교군으로 소량만.** 로컬 처리 우선.
- **패키지 관리는 uv.** `uv add` / `uv run` 만 쓴다. `pip` 직접 사용 금지.

## 구조

```
src/who_approved_this/
  config.py    DATA_ROOT / RESULTS_ROOT
  collect/     소스 어댑터.  base.Collector  → (pdf_path, ground_truth) yield
  parse/       렌더링·OCR·VLM. base.Parser  → pdf_path 받아 str | dict 반환
  evaluate/    대조.          base.Evaluator → (parsed, ground_truth) 받아 점수 dict 반환
  tracks/      트랙별 조합·실행
  cli.py       typer 앱 (`wat run <track>`) — 트랙은 TRACKS dict 에 한 줄 추가
results/       트랙별 점수 JSON
docs/          계획서
```

## 실행

```bash
uv sync
uv run wat run t4
```

## T4 데이터 배치

```
~/data/who-approved-this/t4/
  <이름>.pdf     관보 원문
  <이름>.txt     같은 이름의 정답 전문 (국가법령정보 등 공개 구조화 텍스트)
  _render/       200dpi PNG 중간 산출물 (자동 생성)
```

로컬 OCR은 macOS Vision(`pyobjc-framework-Vision`)을 쓴다. 모델 다운로드가 없고
원문이 로컬 밖으로 나가지 않는다. 리눅스로 옮길 땐 `parse/` 에 다른 `Parser`
구현체를 추가한다 — 골격은 건드리지 않는다.
