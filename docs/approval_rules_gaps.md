# 전결 규칙 스키마 — 빠진 개념과 공개 구현 대비 차이

근거: `DATA_ROOT/research/approval_rules_survey.md` (Aside 조사, 2026-09-11) §8,
그리고 GitHub 원본을 직접 읽은 두 곳 —
[nc2U/ibs `route_builder.py`](https://github.com/nc2U/ibs/blob/master/app/django/approval/services/route_builder.py),
[dalgme/castlog `approvals.sql`](https://github.com/dalgme/castlog) (`supabase/migrations/20260730000002_approvals.sql`,
`20260830000007_approval_grade_relay.sql`).

> 조사의 가장 중요한 한 줄 (§5): **"LLM/ML 로 결재선 자체를 추천하는 검증 가능한 선행 사례는 없다."**
> 시장에 있는 건 전부 룰엔진이다. 우리 T1c·T6 수치는 남의 방법을 따라 한 게 아니라
> 공개 결재문서로 직접 잰 첫 숫자다.

## 1. 빠진 개념 5개 — 어디에 넣었나

엑셀(전결위임표) 시트 수정은 엑셀 실물이 도착한 뒤 한다. 지금은 **yaml 스키마와 코드 자리**만 만들었다.
귀납 규칙(`rules/approval.induced.yaml`, schema v3)에 필드가 들어가 있고, 조회 함수는
`parse/approval_layers.py` 가 읽는다.

| 개념 | yaml 필드 | 코드 | 지금 값 |
|---|---|---|---|
| 우선순위 (동시 매칭) | `delegation.rules[].priority` | `lookup_delegation` — priority **오름차순 첫 매칭 1건** | 귀납은 규칙 1개(priority 1) |
| 미매칭 폴백 | `delegation.fallback {policy, final_title}` | 매칭 없으면 폴백, 결과에 `폴백:chain_top` 표시 | 최상위 결재 직위 |
| 직무대행 (대결자) | `acting[] {slot, by, observed_docs}` | 기록만 — 부재 여부는 문서로 예측 불가 | 경기도서관 대결 1건 관측(10185) |
| 협조 순차/병렬 | `cooperation[].mode` + `mode_provenance` | `lookup_cooperation` 반환에 `mode` | `parallel` / `assumed` (gold 로는 구분 불가) |
| 조인키 | `join_key: "org+title"` | 명부·전결표·결재계층을 (조직, 직위 표준 명칭)으로 잇는다 | — |
| (스텁) 자연어 2열형 금액 구간 | — | `parse/rules_xlsx.amount_bands_from_text` | `NotImplementedError` — 실물 별표 대기 |

priority 방향은 구현마다 반대다(ibs 오름차순, castlog 내림차순 `priority desc`).
우리는 **오름차순(1이 먼저)** 으로 정했다 — 엑셀에서 위 행이 먼저라는 직관과 맞는다.

## 2. 우리 layered 와 다른 점 3개

| | nc2U/ibs | dalgme/castlog | 우리 layered |
|---|---|---|---|
| **① 결재선을 만드는 뼈대** | 기안자 부서에서 **조직도 상향 순회**(부서장 찾기), 전결 직책·부서 레벨에 닿으면 종결 | 규칙 매칭 → `approval_rule_steps` 의 **고정 단계 목록**, 규칙 없으면 직급 에스컬레이션 | **과거 결재선 다수결**이 뼈대, 규칙 층(전결·팀·협조)이 칸을 덮어쓴다. 조직도 순회는 가평군 경로(`slot_chain`)에만 있다 |
| **② 사람을 어떻게 정하나** | 부서의 `manager` 또는 직책 보유자 조회 — **결정론** | `approver_user_id` 를 규칙에 직접 저장 — **결정론** | 직위를 먼저 정하고, 이름은 명부 후보 중 선택. 후보가 여럿이면 **LLM 이 후보 안에서** 고른다 — 확률적인 층이 하나 있다 |
| **③ 운영 규칙 vs 근거 기록** | 셀프 승인 방지, 겸직 시 최상위 직함 승격, 상신 시점 결재선 동결 | 대결 = 기간형 위임(`approval_delegations`), 규칙 버전(`superseded_by_id`), 합의 병렬(`step_kind`, 같은 `step_order`) | 둘 다 없는 것: **칸마다 근거 태그**(roster/rule/vote/llm)와 규칙 **출처**(official/induced/assumed). 반대로 우리엔 셀프 승인 방지·버전·기간형 대결이 없다 |

시사점: 둘 다 "규칙이 정해져 있다"는 전제에서 출발한다. 우리 문제는 규칙이 **불완전하거나 없는**
상태에서 과거 결재선으로 메우는 것이라 뼈대가 다르다. 회사 시스템에 넣을 때는 ②를 ibs·castlog 처럼
결정론으로 바꿀 수 있다(인사 DB 가 사람을 주면 LLM 층이 사라진다). ③의 셀프 승인 방지와
기간형 대결은 그대로 가져올 만하다.
