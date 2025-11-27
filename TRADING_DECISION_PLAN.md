# AI-RAG 암호화폐 트레이딩 시스템 - 거래 판단 계획서

## 1. 시스템 개요

본 시스템은 **AI(Gemini) + RAG(검색 증강 생성) + 기술적 분석**을 결합한 자동 암호화폐 트레이딩 시스템입니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Trading Engine                                │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │
│  │ Market Data  │──▶│ AI Decision  │──▶│    Risk      │            │
│  │   Analyzer   │   │    Engine    │   │   Manager    │            │
│  └──────────────┘   └──────────────┘   └──────────────┘            │
│         │                  │                  │                      │
│         ▼                  ▼                  ▼                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │
│  │  Technical   │   │  RAG Vector  │   │   Position   │            │
│  │  Indicators  │   │    Store     │   │   Sizing     │            │
│  └──────────────┘   └──────────────┘   └──────────────┘            │
│                              │                                       │
│                              ▼                                       │
│                     ┌──────────────┐                                │
│                     │   Binance    │                                │
│                     │   Testnet    │                                │
│                     └──────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 거래 사이클 (Trading Cycle)

### 2.1 메인 루프 (60초 간격)

```python
while is_trading:
    1. 긴급 정지 확인
    2. 시장 데이터 업데이트
    3. 리스크 한도 검사
    4. 기존 포지션 관리
    5. 새로운 기회 탐색
    6. 패턴 수집 (RAG 지식 베이스)
    7. 메트릭 기록
```

### 2.2 각 단계별 상세

| 단계 | 파일 | 함수 | 설명 |
|------|------|------|------|
| 1 | `engine.py` | `_trading_cycle()` | `emergency_stop` 플래그 확인 |
| 2 | `engine.py` | `_update_market_data()` | OHLCV + 기술지표 계산 |
| 3 | `manager.py` | `check_limits()` | 일일 손실, 포지션 수, 총 노출도 |
| 4 | `engine.py` | `_manage_positions()` | 손절/익절/AI 평가 |
| 5 | `engine.py` | `_scan_opportunities()` | AI 시그널 생성 |
| 6 | `pattern_collector.py` | `monitor_patterns()` | RAG 학습 데이터 수집 |

---

## 3. AI 거래 판단 프로세스

### 3.1 시그널 생성 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Signal Generation                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Market Context 준비                                          │
│     ├─ 현재가, 24h 변동률, 거래량                                │
│     ├─ RSI, MACD, EMA 크로스                                    │
│     └─ 지지/저항, 추세, 스프레드                                 │
│                                                                  │
│  2. RAG Query (Vector Store)                                    │
│     ├─ ChromaDB에서 유사 패턴 검색                               │
│     ├─ 과거 거래 결과 참조                                       │
│     └─ 시장 지식 문서 활용                                       │
│                                                                  │
│  3. Gemini Pro 분석 (Primary Model)                             │
│     ├─ Market Context + RAG Insights 전달                       │
│     └─ JSON 형식 결정 생성                                       │
│                                                                  │
│  4. Gemini Flash 검증 (Secondary Model)                         │
│     └─ 빠른 Sanity Check (YES/NO)                               │
│                                                                  │
│  5. Signal 생성 (confidence >= 0.7 인 경우만)                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 AI Decision 구조

```python
@dataclass
class AIDecision:
    action: str        # BUY, SELL, HOLD
    confidence: float  # 0.0 ~ 1.0
    reason: str        # 판단 이유
    entry_price: float # 진입가
    stop_loss: float   # 손절가
    take_profit: float # 익절가
    size: float        # 포지션 크기 (0.01 ~ 0.05)
    risk_score: float  # 위험 점수 (0.0 ~ 1.0)
```

### 3.3 AI 프롬프트 구조

```
You are an expert crypto trader analyzing {symbol}.

Market Context:
- 현재가, 24h 변동률, 거래량
- RSI, MACD, EMA Cross
- 추세, 지지/저항, 스프레드

Historical Patterns & Insights (from RAG):
- 유사한 과거 패턴들
- 이전 거래 결과 및 교훈

Generate a trading decision with:
1. Action: BUY, SELL, or HOLD
2. Confidence: 0.0 to 1.0
3. Entry/Stop Loss/Take Profit
4. Position size
5. Risk score
6. Detailed reasoning
```

---

## 4. 기술적 분석 지표

### 4.1 사용 지표 목록

| 카테고리 | 지표 | 설명 |
|----------|------|------|
| **추세** | SMA 20/50/200 | 단순이동평균 |
| | EMA 12/26/50 | 지수이동평균 |
| | Golden/Death Cross | SMA 50/200 크로스 |
| | ADX | 추세 강도 (25 이상 = 추세 존재) |
| **모멘텀** | RSI | 30 미만 과매도, 70 초과 과매수 |
| | MACD | 시그널 크로스오버 |
| | Stochastic | K/D 라인 크로스 |
| **변동성** | Bollinger Bands | 상/하단 밴드, 스퀴즈 |
| | ATR | 평균 진정 범위 (손절 계산용) |
| **거래량** | OBV | 누적 거래량 |
| | VWAP | 거래량가중평균가 |
| | Volume Ratio | 평균 대비 거래량 비율 |

### 4.2 패턴 감지

```python
# 캔들스틱 패턴
- Doji (도지)
- Hammer (해머)
- Shooting Star (유성)
- Engulfing (장악형)
- Morning/Evening Star (샛별/석별)

# 차트 패턴
- Double Top/Bottom (이중 천장/바닥)
- Uptrend/Downtrend (추세)
- Support/Resistance (지지/저항)
```

### 4.3 시장 구조 분석

```python
market_phase = {
    'ACCUMULATION': '축적 - 낮은 변동성, 증가하는 거래량',
    'MARKUP': '상승 - 강한 상승추세, 좋은 거래량',
    'DISTRIBUTION': '분배 - 고점에서 높은 변동성',
    'MARKDOWN': '하락 - 강한 하락추세'
}
```

---

## 5. 리스크 관리

### 5.1 거래 제한 조건

| 조건 | 임계값 | 액션 |
|------|--------|------|
| 일일 최대 손실 | 5% | 거래 중단 |
| 최대 포지션 수 | 10개 | 신규 진입 금지 |
| 총 노출도 | 50% | 신규 진입 금지 |
| 상관관계 위험 | 0.7 초과 | 포지션 거부 |

### 5.2 포지션 사이징

```python
# Kelly Criterion 적용
kelly_fraction = (win_rate * odds - loss_rate) / odds

# 안전 계수 적용 (Kelly의 25%만 사용)
safe_size = kelly_fraction * 0.25

# 최종 크기 결정
position_size = min(
    signal_size,
    kelly_size,
    max_position_size,
    stop_loss_based_size  # ATR 기반
)
```

### 5.3 손절/익절 전략

```python
# 고정 손절
stop_loss = entry_price * (1 - 0.02)  # 2% 손절

# 트레일링 스톱 (활성화 시)
if current_price > highest_price:
    new_stop = current_price * (1 - 0.03)  # 3% 트레일링
    stop_loss = max(stop_loss, new_stop)

# ATR 기반 손절
atr_stop = entry_price - (2 * ATR)
```

---

## 6. 거래 실행 흐름

### 6.1 진입 조건

```
┌─────────────────────────────────────────────────────────────────┐
│                     Position Entry Flow                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. AI Signal 생성됨 (confidence >= 0.7)                         │
│                         ▼                                        │
│  2. Risk Manager 승인                                            │
│     ├─ 일일 손실 한도 확인                                       │
│     ├─ 상관관계 위험 확인                                        │
│     ├─ Kelly Criterion 포지션 크기 계산                          │
│     └─ 총 노출도 확인                                            │
│                         ▼                                        │
│  3. 잔고 확인 (필요 자본 + 10% 버퍼)                             │
│                         ▼                                        │
│  4. 주문 실행                                                    │
│     ├─ Testnet: Binance Testnet API                             │
│     └─ Paper: PaperTradingEngine (가상)                         │
│                         ▼                                        │
│  5. Position 생성 및 DB 저장                                     │
│                         ▼                                        │
│  6. 알림 전송 (Slack)                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 청산 조건

| 조건 | 트리거 | 설명 |
|------|--------|------|
| 손절 | `price <= stop_loss` | 손실 제한 |
| 익절 | `price >= take_profit` | 이익 실현 |
| AI 결정 | `decision.action == CLOSE` | AI 청산 권고 |
| AI 조정 | `decision.action == ADJUST` | 손절/익절 수정 |
| 긴급 정지 | 5회 연속 에러 | 모든 포지션 청산 |

### 6.3 포지션 관리

```python
for position in open_positions:
    # 1. 현재가 업데이트
    position.update_price(current_price)

    # 2. 손절 확인
    if check_stop_loss(position):
        close_position(position, "Stop loss triggered")
        continue

    # 3. 익절 확인
    if check_take_profit(position):
        close_position(position, "Take profit reached")
        continue

    # 4. AI 평가
    decision = ai_engine.evaluate_position(position)
    if decision.action == "CLOSE":
        close_position(position, decision.reason)
    elif decision.action == "ADJUST":
        adjust_position(position, decision)
```

---

## 7. RAG 지식 베이스

### 7.1 저장되는 지식 유형

```python
knowledge_types = {
    'pattern': '시장 패턴 (강세/약세/횡보)',
    'strategy': '거래 전략 (지지/저항, 거래량)',
    'indicator': '기술 지표 해석 (RSI, MACD)',
    'risk': '리스크 관리 규칙',
    'trade_result': '완료된 거래 결과 및 교훈'
}
```

### 7.2 학습 사이클

```
거래 완료
    ▼
PnL 계산 + 교훈 추출
    ▼
Document 생성
    ▼
ChromaDB에 저장
    ▼
다음 거래 시 RAG로 참조
```

### 7.3 교훈 추출 예시

```python
lessons = []
if pnl > 0:
    lessons.append(f"Successful {side} trade on {symbol}")
    if 'take profit' in reason:
        lessons.append("Take profit target was well-placed")
else:
    lessons.append(f"Losing {side} trade on {symbol}")
    if 'stop loss' in reason:
        lessons.append("Stop loss prevented larger losses")
    else:
        lessons.append("Consider tighter stop loss for similar setups")
```

---

## 8. 현재 설정값

### 8.1 환경 설정 (`.env`)

```bash
# 거래 모드
BINANCE_TESTNET=True        # 테스트넷 사용
PAPER_TRADING=False         # 실제 주문 실행
ENVIRONMENT=production      # 프로덕션 모드

# 리스크 설정
MAX_POSITION_SIZE=0.02      # 포트폴리오의 2%
MAX_DAILY_LOSS=0.05         # 일일 최대 손실 5%
DEFAULT_STOP_LOSS_PERCENT=0.02  # 2% 손절
TRAILING_STOP_PERCENT=0.03  # 3% 트레일링

# Kelly Criterion
KELLY_FRACTION=0.25         # Kelly의 25%만 사용

# AI 설정
AI_CONFIDENCE_THRESHOLD=0.7 # 최소 신뢰도 70%
```

### 8.2 테스트넷 잔고

| 자산 | 수량 |
|------|------|
| USDT | 10,000 |
| BTC | 1 |

---

## 9. 대시보드 모니터링

### 9.1 표시 정보

- **계정 잔고**: USDT, BTC 실시간 잔고
- **시장 차트**: BTC/USDT 캔들스틱 + 지표
- **AI 분석 기록**: 모든 분석 결과 (BUY/SELL/HOLD)
- **포지션 현황**: 열린 포지션 목록
- **거래 내역**: 완료된 거래 기록

### 9.2 접속 URL

```
대시보드: http://152.70.241.36:18501
API: http://152.70.241.36:18080
```

---

## 10. 주의사항

### 10.1 테스트넷 한계

- 실제 시장 유동성과 다름
- 슬리피지가 실제와 다를 수 있음
- 체결 속도가 실제와 다름

### 10.2 실거래 전환 시

```bash
# 실거래로 전환하려면:
BINANCE_TESTNET=False
# 그리고 실제 API 키 설정
```

### 10.3 안전 장치

1. **긴급 정지**: 5회 연속 에러 시 자동 발동
2. **일일 손실 한도**: 5% 초과 시 거래 중단
3. **포지션 제한**: 최대 10개, 총 노출 50%
4. **AI 이중 검증**: Gemini Pro 분석 + Gemini Flash 검증

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2025-11-27 | 1.0 | 초기 문서 작성 |
