# AI 에이전트 기반 가상화폐 자동매매 시스템 - 보완 문서

**작성일:** 2025-11-22
**목적:** PRD 및 구축 계획서에서 누락된 실전 트레이딩 필수 기능 보완

---

## 목차

1. [실전 트레이딩 필수 기능](#1-실전-트레이딩-필수-기능)
2. [주문 실행 엣지 케이스](#2-주문-실행-엣지-케이스)
3. [고급 포지션 관리](#3-고급-포지션-관리)
4. [Paper Trading 완전 구현](#4-paper-trading-완전-구현)
5. [보안 강화 방안](#5-보안-강화-방안)
6. [성능 최적화 전략](#6-성능-최적화-전략)
7. [백테스팅 현실성 개선](#7-백테스팅-현실성-개선)

---

## 1. 실전 트레이딩 필수 기능

### 1.1. 거래소 제약사항 검증 시스템

실제 거래소는 다양한 제약사항을 가지고 있으며, 이를 위반하면 주문이 거부됩니다.

```python
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Tuple, Optional

class ExchangeConstraints:
    """거래소별 제약사항 관리"""

    CONSTRAINTS = {
        'upbit': {
            'KRW-BTC': {
                'min_order_amount': 5000,  # 최소 주문 금액 (KRW)
                'min_order_quantity': 0.0001,  # 최소 주문 수량 (BTC)
                'price_tick': 1000,  # 가격 단위 (KRW)
                'quantity_precision': 8,  # 수량 소수점 자리수
                'max_order_amount': 1000000000,  # 최대 주문 금액
            }
        },
        'binance': {
            'BTC/USDT': {
                'min_order_amount': 10,  # 최소 주문 금액 (USDT)
                'min_order_quantity': 0.00001,  # 최소 주문 수량 (BTC)
                'price_tick': 0.01,  # 가격 단위 (USDT)
                'quantity_precision': 5,
                'max_order_quantity': 9000,  # 최대 주문 수량
            }
        }
    }

    @classmethod
    def validate_order(cls, exchange: str, symbol: str,
                      side: str, quantity: float, price: float) -> Tuple[bool, Optional[str]]:
        """주문 유효성 검증"""
        if exchange not in cls.CONSTRAINTS:
            return False, f"Unknown exchange: {exchange}"

        if symbol not in cls.CONSTRAINTS[exchange]:
            return False, f"Unknown symbol: {symbol} on {exchange}"

        constraints = cls.CONSTRAINTS[exchange][symbol]

        # 최소 주문 금액 검증
        order_amount = quantity * price
        if order_amount < constraints['min_order_amount']:
            return False, f"Order amount {order_amount} below minimum {constraints['min_order_amount']}"

        # 최소 주문 수량 검증
        if quantity < constraints['min_order_quantity']:
            return False, f"Quantity {quantity} below minimum {constraints['min_order_quantity']}"

        # 최대 제한 검증
        if 'max_order_amount' in constraints and order_amount > constraints['max_order_amount']:
            return False, f"Order amount {order_amount} exceeds maximum {constraints['max_order_amount']}"

        if 'max_order_quantity' in constraints and quantity > constraints['max_order_quantity']:
            return False, f"Quantity {quantity} exceeds maximum {constraints['max_order_quantity']}"

        return True, None

    @classmethod
    def adjust_price(cls, exchange: str, symbol: str, price: float) -> Decimal:
        """가격을 거래소 틱 단위로 조정"""
        constraints = cls.CONSTRAINTS[exchange][symbol]
        tick = Decimal(str(constraints['price_tick']))
        price_decimal = Decimal(str(price))

        # 틱 단위로 반올림
        adjusted = (price_decimal / tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * tick
        return adjusted

    @classmethod
    def adjust_quantity(cls, exchange: str, symbol: str, quantity: float) -> Decimal:
        """수량을 거래소 정밀도로 조정"""
        constraints = cls.CONSTRAINTS[exchange][symbol]
        precision = constraints['quantity_precision']

        # 정밀도에 맞춰 버림
        quantize_str = '0.' + '0' * (precision - 1) + '1'
        quantity_decimal = Decimal(str(quantity))
        adjusted = quantity_decimal.quantize(Decimal(quantize_str), rounding=ROUND_DOWN)

        return adjusted
```

### 1.2. 부분 체결 처리 시스템

대량 주문이나 유동성이 낮은 시장에서는 주문이 부분적으로만 체결될 수 있습니다.

```python
import asyncio
from enum import Enum
from typing import Dict, Optional

class OrderStatus(Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"

class PartialFillHandler:
    """부분 체결 처리 관리자"""

    def __init__(self, exchange_client):
        self.exchange = exchange_client
        self.active_orders = {}  # order_id -> order_info

    async def execute_with_partial_handling(self,
                                           symbol: str,
                                           side: str,
                                           quantity: float,
                                           price: Optional[float] = None,
                                           timeout: int = 300) -> Dict:
        """부분 체결을 고려한 주문 실행"""

        # 1. 초기 주문 발생
        if price:
            order = await self.exchange.create_limit_order(symbol, side, quantity, price)
        else:
            order = await self.exchange.create_market_order(symbol, side, quantity)

        order_id = order['id']
        self.active_orders[order_id] = {
            'symbol': symbol,
            'side': side,
            'original_quantity': quantity,
            'filled_quantity': 0,
            'remaining_quantity': quantity,
            'status': OrderStatus.PENDING,
            'fills': []  # 체결 내역
        }

        # 2. 체결 모니터링
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            # 주문 상태 확인
            order_status = await self.exchange.fetch_order(order_id, symbol)

            filled = order_status.get('filled', 0)
            remaining = order_status.get('remaining', quantity)
            status = order_status.get('status', 'open')

            # 상태 업데이트
            self.active_orders[order_id]['filled_quantity'] = filled
            self.active_orders[order_id]['remaining_quantity'] = remaining

            if status == 'closed':
                # 완전 체결
                self.active_orders[order_id]['status'] = OrderStatus.FILLED
                logger.info(f"Order {order_id} fully filled: {filled} {symbol}")
                return self.active_orders[order_id]

            elif filled > 0 and remaining > 0:
                # 부분 체결
                self.active_orders[order_id]['status'] = OrderStatus.PARTIAL
                logger.info(f"Order {order_id} partially filled: {filled}/{quantity} {symbol}")

                # 부분 체결 처리 전략 결정
                action = await self._decide_partial_fill_action(
                    order_id, filled, remaining, price
                )

                if action == 'wait':
                    # 추가 체결 대기
                    await asyncio.sleep(5)
                    continue

                elif action == 'cancel_and_market':
                    # 미체결 수량 취소 후 시장가 주문
                    await self.exchange.cancel_order(order_id, symbol)

                    if remaining > 0:
                        market_order = await self.exchange.create_market_order(
                            symbol, side, remaining
                        )
                        logger.info(f"Remaining {remaining} executed at market")

                        # 최종 결과 병합
                        self.active_orders[order_id]['fills'].append(market_order)
                        self.active_orders[order_id]['status'] = OrderStatus.FILLED
                        return self.active_orders[order_id]

                elif action == 'cancel':
                    # 미체결 수량 취소
                    await self.exchange.cancel_order(order_id, symbol)
                    self.active_orders[order_id]['status'] = OrderStatus.PARTIAL
                    return self.active_orders[order_id]

            await asyncio.sleep(1)

        # 타임아웃 - 미체결 수량 취소
        try:
            await self.exchange.cancel_order(order_id, symbol)
            logger.warning(f"Order {order_id} timeout after {timeout}s")
        except:
            pass

        return self.active_orders[order_id]

    async def _decide_partial_fill_action(self, order_id: str,
                                         filled: float,
                                         remaining: float,
                                         target_price: float) -> str:
        """부분 체결 시 행동 결정"""

        # 전략 1: 80% 이상 체결되면 나머지는 시장가
        if filled / (filled + remaining) >= 0.8:
            return 'cancel_and_market'

        # 전략 2: 시장 가격이 목표가에서 크게 벗어나면 취소
        current_price = await self._get_current_price(
            self.active_orders[order_id]['symbol']
        )

        price_deviation = abs(current_price - target_price) / target_price
        if price_deviation > 0.02:  # 2% 이상 벗어남
            return 'cancel'

        # 전략 3: 그 외에는 대기
        return 'wait'

    async def _get_current_price(self, symbol: str) -> float:
        """현재 시장가 조회"""
        ticker = await self.exchange.fetch_ticker(symbol)
        return ticker['last']
```

### 1.3. 슬리피지 예측 및 관리

실제 거래에서는 주문 가격과 체결 가격 간 차이(슬리피지)가 발생합니다.

```python
import numpy as np
from typing import Dict, Tuple

class SlippageManager:
    """슬리피지 예측 및 관리"""

    def __init__(self):
        self.slippage_history = []  # 과거 슬리피지 기록
        self.volatility_threshold = 0.02  # 2% 변동성 임계값

    async def estimate_slippage(self,
                               symbol: str,
                               side: str,
                               quantity: float,
                               exchange_client) -> Dict:
        """슬리피지 예측"""

        # 1. 호가창 분석
        orderbook = await exchange_client.fetch_order_book(symbol, limit=50)

        # 2. 시장 깊이 기반 슬리피지 계산
        if side == 'BUY':
            orders = orderbook['asks']
        else:
            orders = orderbook['bids']

        cumulative_volume = 0
        weighted_price_sum = 0

        for price, volume in orders:
            if cumulative_volume + volume >= quantity:
                # 이 호가에서 체결 완료
                remaining = quantity - cumulative_volume
                weighted_price_sum += price * remaining
                cumulative_volume = quantity
                break
            else:
                weighted_price_sum += price * volume
                cumulative_volume += volume

        if cumulative_volume < quantity:
            # 호가창 깊이 부족
            return {
                'estimated_slippage': None,
                'error': 'Insufficient orderbook depth',
                'available_volume': cumulative_volume
            }

        # 3. 평균 체결 예상가 계산
        avg_execution_price = weighted_price_sum / quantity
        best_price = orders[0][0]
        slippage_pct = abs(avg_execution_price - best_price) / best_price * 100

        # 4. 변동성 조정
        volatility = await self._calculate_volatility(symbol, exchange_client)
        adjusted_slippage = slippage_pct * (1 + volatility)

        return {
            'best_price': best_price,
            'estimated_avg_price': avg_execution_price,
            'estimated_slippage_pct': adjusted_slippage,
            'estimated_cost': avg_execution_price * quantity,
            'market_impact': self._estimate_market_impact(quantity, cumulative_volume)
        }

    async def _calculate_volatility(self, symbol: str, exchange_client) -> float:
        """최근 변동성 계산"""
        # 최근 1시간 캔들 데이터
        candles = await exchange_client.fetch_ohlcv(symbol, '1m', limit=60)

        # 수익률 계산
        closes = [candle[4] for candle in candles]
        returns = np.diff(np.log(closes))

        # 변동성 (표준편차)
        volatility = np.std(returns)
        return volatility

    def _estimate_market_impact(self, order_size: float, market_depth: float) -> str:
        """시장 충격도 평가"""
        impact_ratio = order_size / market_depth

        if impact_ratio < 0.01:
            return "negligible"
        elif impact_ratio < 0.05:
            return "low"
        elif impact_ratio < 0.1:
            return "moderate"
        else:
            return "high"

    def record_actual_slippage(self,
                              expected_price: float,
                              actual_price: float,
                              quantity: float):
        """실제 슬리피지 기록"""
        slippage_pct = abs(actual_price - expected_price) / expected_price * 100

        self.slippage_history.append({
            'timestamp': datetime.utcnow(),
            'expected_price': expected_price,
            'actual_price': actual_price,
            'quantity': quantity,
            'slippage_pct': slippage_pct
        })

        # 최근 100개 기록만 유지
        if len(self.slippage_history) > 100:
            self.slippage_history = self.slippage_history[-100:]

    def get_average_slippage(self) -> float:
        """평균 슬리피지 계산"""
        if not self.slippage_history:
            return 0

        return np.mean([h['slippage_pct'] for h in self.slippage_history])
```

---

## 2. 주문 실행 엣지 케이스

### 2.1. 거래소 점검 및 장애 처리

```python
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional

class ExchangeStatusMonitor:
    """거래소 상태 모니터링 및 장애 처리"""

    def __init__(self):
        self.exchange_status = {}
        self.maintenance_schedule = {}
        self.last_health_check = {}

    async def monitor_exchange_health(self, exchange_name: str, exchange_client):
        """거래소 상태 실시간 모니터링"""

        while True:
            try:
                # 1. API 상태 확인
                start_time = asyncio.get_event_loop().time()
                status = await exchange_client.fetch_status()
                latency = (asyncio.get_event_loop().time() - start_time) * 1000  # ms

                # 2. 상태 업데이트
                self.exchange_status[exchange_name] = {
                    'status': status.get('status', 'unknown'),
                    'latency_ms': latency,
                    'last_check': datetime.utcnow(),
                    'consecutive_failures': 0
                }

                # 3. 지연시간 경고
                if latency > 1000:  # 1초 초과
                    logger.warning(f"{exchange_name} high latency: {latency}ms")
                    await self._notify_high_latency(exchange_name, latency)

                # 4. 점검 공지 확인
                maintenance = await self._check_maintenance_notice(exchange_name)
                if maintenance:
                    self.maintenance_schedule[exchange_name] = maintenance
                    await self._prepare_for_maintenance(exchange_name, maintenance)

            except Exception as e:
                # 연결 실패
                if exchange_name not in self.exchange_status:
                    self.exchange_status[exchange_name] = {'consecutive_failures': 0}

                self.exchange_status[exchange_name]['consecutive_failures'] += 1
                failures = self.exchange_status[exchange_name]['consecutive_failures']

                logger.error(f"{exchange_name} health check failed ({failures}x): {e}")

                # 3회 연속 실패 시 거래 중단
                if failures >= 3:
                    await self._handle_exchange_outage(exchange_name)

            await asyncio.sleep(30)  # 30초마다 확인

    async def _handle_exchange_outage(self, exchange_name: str):
        """거래소 장애 처리"""
        logger.critical(f"{exchange_name} is down! Initiating emergency procedures")

        # 1. 모든 미체결 주문 취소 시도
        try:
            await self._cancel_all_open_orders(exchange_name)
        except:
            pass

        # 2. 포지션 보호 (가능한 경우 다른 거래소로 헤지)
        await self._protect_positions(exchange_name)

        # 3. 긴급 알림 발송
        await self._send_emergency_alert(
            f"🚨 {exchange_name} 거래소 장애 발생!\n"
            f"모든 거래가 중단되었습니다.\n"
            f"수동 확인이 필요합니다."
        )

        # 4. 거래 봇 일시 정지
        await self._pause_trading(exchange_name)

    async def _check_maintenance_notice(self, exchange_name: str) -> Optional[Dict]:
        """거래소 점검 공지 확인"""
        # 거래소별 공지 API 또는 웹 스크래핑
        # 예시: Upbit의 경우
        if exchange_name == 'upbit':
            # Upbit 공지사항 API 또는 스크래핑
            pass

        return None

    async def _prepare_for_maintenance(self, exchange_name: str, maintenance: Dict):
        """점검 대비"""
        start_time = maintenance['start_time']
        end_time = maintenance['end_time']

        # 점검 1시간 전부터 신규 포지션 진입 금지
        if datetime.utcnow() >= start_time - timedelta(hours=1):
            logger.warning(f"{exchange_name} maintenance in 1 hour. Stopping new positions")
            await self._stop_new_positions(exchange_name)

        # 점검 10분 전 모든 포지션 청산
        if datetime.utcnow() >= start_time - timedelta(minutes=10):
            logger.warning(f"{exchange_name} maintenance in 10 min. Closing all positions")
            await self._close_all_positions(exchange_name)
```

### 2.2. 서킷브레이커 대응

```python
class CircuitBreakerHandler:
    """거래 중단(서킷브레이커) 대응"""

    def __init__(self):
        self.circuit_breaker_active = {}
        self.price_before_halt = {}

    async def detect_circuit_breaker(self, symbol: str, exchange_client) -> bool:
        """서킷브레이커 감지"""

        # 1. 거래량 급감 확인
        current_volume = await self._get_recent_volume(symbol, exchange_client)
        avg_volume = await self._get_average_volume(symbol, exchange_client)

        if current_volume < avg_volume * 0.1:  # 평균 대비 10% 미만
            # 2. 호가창 확인
            orderbook = await exchange_client.fetch_order_book(symbol)

            # 호가가 없거나 스프레드가 비정상적으로 넓음
            if not orderbook['bids'] or not orderbook['asks']:
                return True

            spread = orderbook['asks'][0][0] - orderbook['bids'][0][0]
            mid_price = (orderbook['asks'][0][0] + orderbook['bids'][0][0]) / 2
            spread_pct = spread / mid_price * 100

            if spread_pct > 5:  # 스프레드 5% 초과
                return True

        return False

    async def handle_circuit_breaker(self, symbol: str, exchange_client):
        """서킷브레이커 대응"""

        logger.critical(f"Circuit breaker detected for {symbol}")

        # 1. 현재 가격 저장
        ticker = await exchange_client.fetch_ticker(symbol)
        self.price_before_halt[symbol] = ticker['last']

        # 2. 거래 중단 플래그 설정
        self.circuit_breaker_active[symbol] = True

        # 3. 모든 미체결 주문 취소
        await self._cancel_all_orders_for_symbol(symbol, exchange_client)

        # 4. 거래 재개 모니터링
        asyncio.create_task(
            self._monitor_trading_resumption(symbol, exchange_client)
        )

        # 5. 알림 발송
        await self._notify_circuit_breaker(symbol)

    async def _monitor_trading_resumption(self, symbol: str, exchange_client):
        """거래 재개 모니터링"""

        while self.circuit_breaker_active.get(symbol, False):
            # 거래량 회복 확인
            if not await self.detect_circuit_breaker(symbol, exchange_client):
                logger.info(f"Trading resumed for {symbol}")

                # 재개 후 가격 확인
                ticker = await exchange_client.fetch_ticker(symbol)
                current_price = ticker['last']
                halt_price = self.price_before_halt.get(symbol, current_price)

                gap_pct = (current_price - halt_price) / halt_price * 100

                if abs(gap_pct) > 10:
                    logger.warning(f"Large gap after resumption: {gap_pct:.2f}%")
                    # 포지션 재평가 필요
                    await self._reevaluate_positions(symbol, gap_pct)

                self.circuit_breaker_active[symbol] = False
                break

            await asyncio.sleep(10)
```

---

## 3. 고급 포지션 관리

### 3.1. 평균 단가 추적 시스템

```python
from decimal import Decimal
from typing import Dict, List
import json

class PositionTracker:
    """포지션 추적 및 관리"""

    def __init__(self, db_connection):
        self.db = db_connection
        self.positions = {}  # symbol -> position_info

    def add_trade(self, symbol: str, side: str, quantity: float,
                  price: float, commission: float = 0) -> Dict:
        """거래 추가 및 포지션 업데이트"""

        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': Decimal('0'),
                'avg_entry_price': Decimal('0'),
                'realized_pnl': Decimal('0'),
                'trades': []
            }

        pos = self.positions[symbol]
        quantity_decimal = Decimal(str(quantity))
        price_decimal = Decimal(str(price))
        commission_decimal = Decimal(str(commission))

        if side == 'BUY':
            # 매수 - 평균 단가 재계산
            total_cost = pos['quantity'] * pos['avg_entry_price']
            new_cost = quantity_decimal * price_decimal + commission_decimal

            new_quantity = pos['quantity'] + quantity_decimal
            if new_quantity > 0:
                pos['avg_entry_price'] = (total_cost + new_cost) / new_quantity

            pos['quantity'] = new_quantity

        elif side == 'SELL':
            # 매도 - 실현 손익 계산
            if pos['quantity'] <= 0:
                logger.error(f"Cannot sell {symbol}: no position")
                return None

            # 실현 손익 = (매도가 - 평균매수가) * 수량 - 수수료
            pnl = (price_decimal - pos['avg_entry_price']) * quantity_decimal - commission_decimal
            pos['realized_pnl'] += pnl

            # 수량 감소
            pos['quantity'] -= quantity_decimal

            # 포지션 청산 시 초기화
            if pos['quantity'] == 0:
                pos['avg_entry_price'] = Decimal('0')

        # 거래 기록
        trade_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'side': side,
            'quantity': str(quantity_decimal),
            'price': str(price_decimal),
            'commission': str(commission_decimal),
            'position_after': {
                'quantity': str(pos['quantity']),
                'avg_price': str(pos['avg_entry_price']),
                'realized_pnl': str(pos['realized_pnl'])
            }
        }

        pos['trades'].append(trade_record)

        # DB 저장
        self._save_position_to_db(symbol, pos)

        return trade_record

    def get_unrealized_pnl(self, symbol: str, current_price: float) -> Decimal:
        """미실현 손익 계산"""
        if symbol not in self.positions:
            return Decimal('0')

        pos = self.positions[symbol]
        if pos['quantity'] <= 0:
            return Decimal('0')

        current_price_decimal = Decimal(str(current_price))
        unrealized_pnl = (current_price_decimal - pos['avg_entry_price']) * pos['quantity']

        return unrealized_pnl

    def get_position_summary(self, current_prices: Dict[str, float]) -> Dict:
        """전체 포지션 요약"""
        summary = {
            'positions': [],
            'total_value': Decimal('0'),
            'total_unrealized_pnl': Decimal('0'),
            'total_realized_pnl': Decimal('0')
        }

        for symbol, pos in self.positions.items():
            if pos['quantity'] > 0:
                current_price = Decimal(str(current_prices.get(symbol, 0)))
                unrealized_pnl = self.get_unrealized_pnl(symbol, float(current_price))
                position_value = pos['quantity'] * current_price

                summary['positions'].append({
                    'symbol': symbol,
                    'quantity': str(pos['quantity']),
                    'avg_entry_price': str(pos['avg_entry_price']),
                    'current_price': str(current_price),
                    'position_value': str(position_value),
                    'unrealized_pnl': str(unrealized_pnl),
                    'unrealized_pnl_pct': str((unrealized_pnl / (pos['quantity'] * pos['avg_entry_price']) * 100) if pos['avg_entry_price'] > 0 else 0),
                    'realized_pnl': str(pos['realized_pnl'])
                })

                summary['total_value'] += position_value
                summary['total_unrealized_pnl'] += unrealized_pnl
                summary['total_realized_pnl'] += pos['realized_pnl']

        summary['total_value'] = str(summary['total_value'])
        summary['total_unrealized_pnl'] = str(summary['total_unrealized_pnl'])
        summary['total_realized_pnl'] = str(summary['total_realized_pnl'])

        return summary

    def _save_position_to_db(self, symbol: str, position_data: Dict):
        """포지션 데이터 DB 저장"""
        cursor = self.db.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO positions
            (symbol, quantity, avg_entry_price, realized_pnl, trades_json, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            symbol,
            str(position_data['quantity']),
            str(position_data['avg_entry_price']),
            str(position_data['realized_pnl']),
            json.dumps(position_data['trades'])
        ))

        self.db.commit()
```

### 3.2. 포트폴리오 리밸런싱

```python
from typing import Dict, List, Tuple
import numpy as np

class PortfolioRebalancer:
    """포트폴리오 리밸런싱 관리"""

    def __init__(self, target_allocations: Dict[str, float]):
        """
        target_allocations: {'BTC': 0.4, 'ETH': 0.3, 'BNB': 0.2, 'CASH': 0.1}
        """
        self.target_allocations = target_allocations
        self.rebalance_threshold = 0.05  # 5% 이상 벗어나면 리밸런싱
        self.min_trade_size = 100  # 최소 거래 금액 (USD)

    def calculate_rebalance_trades(self,
                                  current_positions: Dict[str, float],
                                  current_prices: Dict[str, float],
                                  total_value: float) -> List[Dict]:
        """리밸런싱 필요 거래 계산"""

        trades = []

        # 1. 현재 배분 비율 계산
        current_allocations = {}
        for symbol, quantity in current_positions.items():
            if symbol in current_prices:
                value = quantity * current_prices[symbol]
                current_allocations[symbol] = value / total_value

        # 현금 비율
        cash_value = total_value - sum(
            current_positions.get(s, 0) * current_prices.get(s, 0)
            for s in current_positions
        )
        current_allocations['CASH'] = cash_value / total_value

        # 2. 리밸런싱 필요 여부 확인
        needs_rebalance = False
        for symbol, target in self.target_allocations.items():
            current = current_allocations.get(symbol, 0)
            deviation = abs(current - target)

            if deviation > self.rebalance_threshold:
                needs_rebalance = True
                break

        if not needs_rebalance:
            return []

        # 3. 목표 포지션 계산
        target_values = {
            symbol: total_value * allocation
            for symbol, allocation in self.target_allocations.items()
            if symbol != 'CASH'
        }

        # 4. 필요 거래 계산
        for symbol, target_value in target_values.items():
            if symbol not in current_prices:
                continue

            current_value = current_positions.get(symbol, 0) * current_prices[symbol]
            diff_value = target_value - current_value

            if abs(diff_value) < self.min_trade_size:
                continue  # 최소 거래 금액 미달

            diff_quantity = diff_value / current_prices[symbol]

            trades.append({
                'symbol': symbol,
                'side': 'BUY' if diff_quantity > 0 else 'SELL',
                'quantity': abs(diff_quantity),
                'reason': 'rebalance',
                'current_allocation': current_allocations.get(symbol, 0),
                'target_allocation': self.target_allocations[symbol]
            })

        # 5. 거래 우선순위 정렬 (편차가 큰 것부터)
        trades.sort(key=lambda x: abs(x['current_allocation'] - x['target_allocation']), reverse=True)

        return trades

    def calculate_portfolio_metrics(self,
                                   positions_history: List[Dict],
                                   prices_history: List[Dict]) -> Dict:
        """포트폴리오 성과 지표 계산"""

        # 일별 수익률 계산
        returns = []
        for i in range(1, len(positions_history)):
            prev_value = self._calculate_portfolio_value(
                positions_history[i-1], prices_history[i-1]
            )
            curr_value = self._calculate_portfolio_value(
                positions_history[i], prices_history[i]
            )

            daily_return = (curr_value - prev_value) / prev_value if prev_value > 0 else 0
            returns.append(daily_return)

        returns_array = np.array(returns)

        # 성과 지표
        metrics = {
            'total_return': np.prod(1 + returns_array) - 1,
            'annualized_return': (1 + np.mean(returns_array)) ** 365 - 1,
            'volatility': np.std(returns_array) * np.sqrt(365),
            'sharpe_ratio': np.mean(returns_array) / np.std(returns_array) * np.sqrt(365) if np.std(returns_array) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(returns_array),
            'win_rate': np.sum(returns_array > 0) / len(returns_array) if len(returns_array) > 0 else 0,
            'avg_win': np.mean(returns_array[returns_array > 0]) if np.any(returns_array > 0) else 0,
            'avg_loss': np.mean(returns_array[returns_array < 0]) if np.any(returns_array < 0) else 0
        }

        return metrics

    def _calculate_portfolio_value(self, positions: Dict, prices: Dict) -> float:
        """포트폴리오 가치 계산"""
        value = 0
        for symbol, quantity in positions.items():
            if symbol in prices:
                value += quantity * prices[symbol]
        return value

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """최대 낙폭 계산"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
```

---

## 4. Paper Trading 완전 구현

### 4.1. Paper Trading 엔진

```python
from typing import Dict, Optional, List
import uuid
from datetime import datetime

class PaperTradingEngine:
    """모의 거래 엔진"""

    def __init__(self, initial_balance: float = 10000):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}  # symbol -> quantity
        self.orders = {}  # order_id -> order_info
        self.trades = []  # 체결 내역
        self.order_id_counter = 0

    async def create_order(self,
                          symbol: str,
                          side: str,
                          order_type: str,
                          quantity: float,
                          price: Optional[float] = None,
                          real_market_data: Dict = None) -> Dict:
        """모의 주문 생성"""

        order_id = f"PAPER_{self.order_id_counter}"
        self.order_id_counter += 1

        # 현재 시장가 (실제 데이터 사용)
        market_price = real_market_data.get('price', 0) if real_market_data else 0

        order = {
            'id': order_id,
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity,
            'price': price if order_type == 'limit' else market_price,
            'status': 'pending',
            'created_at': datetime.utcnow(),
            'filled_quantity': 0,
            'avg_fill_price': 0
        }

        self.orders[order_id] = order

        # 즉시 체결 시뮬레이션 (시장가 또는 조건 충족 시)
        if order_type == 'market':
            await self._execute_order(order_id, market_price, quantity)
        elif order_type == 'limit':
            # 지정가 주문은 조건 확인
            if (side == 'BUY' and market_price <= price) or \
               (side == 'SELL' and market_price >= price):
                await self._execute_order(order_id, price, quantity)

        return order

    async def _execute_order(self, order_id: str, fill_price: float, fill_quantity: float):
        """주문 체결 시뮬레이션"""

        order = self.orders[order_id]

        # 잔고 확인
        if order['side'] == 'BUY':
            required_balance = fill_price * fill_quantity
            if self.balance < required_balance:
                order['status'] = 'rejected'
                order['reject_reason'] = 'Insufficient balance'
                return

            # 매수 실행
            self.balance -= required_balance

            if order['symbol'] not in self.positions:
                self.positions[order['symbol']] = 0
            self.positions[order['symbol']] += fill_quantity

        elif order['side'] == 'SELL':
            # 보유 수량 확인
            if order['symbol'] not in self.positions or \
               self.positions[order['symbol']] < fill_quantity:
                order['status'] = 'rejected'
                order['reject_reason'] = 'Insufficient position'
                return

            # 매도 실행
            self.positions[order['symbol']] -= fill_quantity
            self.balance += fill_price * fill_quantity

            if self.positions[order['symbol']] == 0:
                del self.positions[order['symbol']]

        # 주문 상태 업데이트
        order['status'] = 'filled'
        order['filled_quantity'] = fill_quantity
        order['avg_fill_price'] = fill_price
        order['filled_at'] = datetime.utcnow()

        # 체결 기록
        trade = {
            'trade_id': str(uuid.uuid4()),
            'order_id': order_id,
            'symbol': order['symbol'],
            'side': order['side'],
            'quantity': fill_quantity,
            'price': fill_price,
            'timestamp': datetime.utcnow()
        }

        self.trades.append(trade)

        logger.info(f"Paper trade executed: {trade}")

    async def cancel_order(self, order_id: str) -> bool:
        """주문 취소"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order['status'] == 'pending':
                order['status'] = 'cancelled'
                return True
        return False

    def get_account_info(self) -> Dict:
        """계좌 정보 조회"""
        total_value = self.balance

        # 포지션 평가액 (실시간 가격 필요)
        # 여기서는 간단히 balance만 반환

        return {
            'balance': self.balance,
            'initial_balance': self.initial_balance,
            'positions': self.positions.copy(),
            'total_trades': len(self.trades),
            'pnl': self.balance - self.initial_balance,
            'pnl_percentage': ((self.balance - self.initial_balance) / self.initial_balance * 100)
        }

    def get_performance_metrics(self, current_prices: Dict[str, float]) -> Dict:
        """성과 지표 계산"""

        # 포지션 평가액 계산
        position_value = sum(
            quantity * current_prices.get(symbol, 0)
            for symbol, quantity in self.positions.items()
        )

        total_value = self.balance + position_value

        # 승률 계산
        winning_trades = [t for t in self.trades if t['side'] == 'SELL']  # 간단 예시

        return {
            'total_value': total_value,
            'cash_balance': self.balance,
            'position_value': position_value,
            'total_return': (total_value - self.initial_balance) / self.initial_balance * 100,
            'num_trades': len(self.trades),
            'win_rate': len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        }
```

### 4.2. Paper/Real Trading 전환

```python
class TradingModeManager:
    """거래 모드 관리 (Paper/Real)"""

    def __init__(self, config: Dict):
        self.mode = config.get('trading_mode', 'paper')  # 'paper' or 'real'
        self.paper_engine = PaperTradingEngine(
            initial_balance=config.get('paper_initial_balance', 10000)
        )
        self.real_exchange = None
        self.transition_criteria = config.get('transition_criteria', {})

    def set_mode(self, mode: str):
        """거래 모드 설정"""
        if mode not in ['paper', 'real']:
            raise ValueError("Mode must be 'paper' or 'real'")

        if mode == 'real' and self.mode == 'paper':
            # Paper → Real 전환 조건 확인
            if not self._validate_transition_to_real():
                raise ValueError("Not ready for real trading. Check transition criteria.")

        self.mode = mode
        logger.info(f"Trading mode set to: {mode}")

    def _validate_transition_to_real(self) -> bool:
        """실전 전환 조건 검증"""

        metrics = self.paper_engine.get_performance_metrics({})

        # 전환 조건
        criteria = {
            'min_trades': 100,  # 최소 100회 거래
            'min_win_rate': 55,  # 최소 55% 승률
            'max_drawdown': -20,  # 최대 낙폭 -20% 이내
            'min_days': 30  # 최소 30일 운영
        }

        # 조건 확인
        if metrics['num_trades'] < criteria['min_trades']:
            logger.warning(f"Insufficient trades: {metrics['num_trades']} < {criteria['min_trades']}")
            return False

        if metrics['win_rate'] < criteria['min_win_rate']:
            logger.warning(f"Low win rate: {metrics['win_rate']} < {criteria['min_win_rate']}")
            return False

        # 추가 조건들...

        return True

    async def execute_trade(self, decision: Dict) -> Dict:
        """모드에 따른 거래 실행"""

        if self.mode == 'paper':
            # Paper Trading
            result = await self.paper_engine.create_order(
                symbol=decision['symbol'],
                side=decision['action'],
                order_type='market',
                quantity=decision['quantity'],
                real_market_data={'price': decision.get('current_price', 0)}
            )

            # Paper 거래 결과를 실제와 유사하게 포맷
            return {
                'order_id': result['id'],
                'status': result['status'],
                'filled_quantity': result.get('filled_quantity', 0),
                'avg_price': result.get('avg_fill_price', 0),
                'is_paper': True
            }

        else:
            # Real Trading
            result = await self.real_exchange.create_order(
                symbol=decision['symbol'],
                side=decision['action'],
                type='market',
                amount=decision['quantity']
            )

            return {
                'order_id': result['id'],
                'status': result['status'],
                'filled_quantity': result['filled'],
                'avg_price': result['average'],
                'is_paper': False
            }

    def get_account_info(self) -> Dict:
        """계좌 정보 조회"""
        if self.mode == 'paper':
            info = self.paper_engine.get_account_info()
            info['mode'] = 'PAPER'
        else:
            # 실제 계좌 정보
            balance = self.real_exchange.fetch_balance()
            info = {
                'balance': balance['USDT']['free'],
                'positions': balance,
                'mode': 'REAL'
            }

        return info
```

---

## 5. 보안 강화 방안

### 5.1. API 키 보안 관리

```python
import os
import json
from cryptography.fernet import Fernet
from typing import Dict, Optional

class SecureAPIManager:
    """API 키 암호화 및 관리"""

    def __init__(self, master_key_path: str = '.master_key'):
        self.master_key_path = master_key_path
        self.cipher = self._load_or_create_cipher()
        self.api_keys = {}
        self.permissions = {}

    def _load_or_create_cipher(self) -> Fernet:
        """마스터 키 로드 또는 생성"""
        if os.path.exists(self.master_key_path):
            with open(self.master_key_path, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self.master_key_path, 'wb') as f:
                f.write(key)
            os.chmod(self.master_key_path, 0o600)  # 소유자만 읽기 가능

        return Fernet(key)

    def add_api_key(self,
                    service: str,
                    api_key: str,
                    secret_key: str,
                    permissions: List[str]) -> bool:
        """API 키 추가 (암호화)"""

        # 권한 검증
        valid_permissions = ['read', 'trade', 'withdraw']
        for perm in permissions:
            if perm not in valid_permissions:
                raise ValueError(f"Invalid permission: {perm}")

        # 암호화
        encrypted_api = self.cipher.encrypt(api_key.encode())
        encrypted_secret = self.cipher.encrypt(secret_key.encode())

        # 저장
        self.api_keys[service] = {
            'api_key': encrypted_api,
            'secret_key': encrypted_secret,
            'permissions': permissions,
            'created_at': datetime.utcnow().isoformat()
        }

        # 파일에 저장
        self._save_keys()

        logger.info(f"API key added for {service} with permissions: {permissions}")
        return True

    def get_api_key(self, service: str, required_permission: str = 'read') -> Dict:
        """API 키 조회 (복호화)"""

        if service not in self.api_keys:
            raise ValueError(f"No API key for service: {service}")

        key_info = self.api_keys[service]

        # 권한 확인
        if required_permission not in key_info['permissions']:
            raise PermissionError(
                f"API key for {service} lacks {required_permission} permission"
            )

        # 복호화
        return {
            'api_key': self.cipher.decrypt(key_info['api_key']).decode(),
            'secret_key': self.cipher.decrypt(key_info['secret_key']).decode(),
            'permissions': key_info['permissions']
        }

    def rotate_api_key(self, service: str, new_api_key: str, new_secret_key: str):
        """API 키 순환"""

        if service not in self.api_keys:
            raise ValueError(f"No API key for service: {service}")

        old_permissions = self.api_keys[service]['permissions']

        # 이전 키 백업
        backup = {
            'service': service,
            'old_key': self.api_keys[service],
            'rotated_at': datetime.utcnow().isoformat()
        }

        with open(f'.api_key_backup_{service}_{datetime.utcnow().strftime("%Y%m%d")}.json', 'w') as f:
            json.dump(backup, f)

        # 새 키로 교체
        self.add_api_key(service, new_api_key, new_secret_key, old_permissions)

        logger.info(f"API key rotated for {service}")

    def _save_keys(self):
        """암호화된 키 파일 저장"""
        with open('.encrypted_api_keys.json', 'w') as f:
            json.dump({
                k: {
                    'api_key': v['api_key'].decode('latin-1'),
                    'secret_key': v['secret_key'].decode('latin-1'),
                    'permissions': v['permissions'],
                    'created_at': v['created_at']
                }
                for k, v in self.api_keys.items()
            }, f)

        os.chmod('.encrypted_api_keys.json', 0o600)

    def set_ip_whitelist(self, service: str, ip_addresses: List[str]):
        """IP 화이트리스트 설정 (거래소 API)"""
        # 거래소 API에 IP 제한 설정
        # 실제 구현은 거래소 API에 따라 다름
        pass
```

### 5.2. 강화된 Kill Switch

```python
import signal
import sys
from typing import Callable

class EnhancedKillSwitch:
    """강화된 긴급 정지 시스템"""

    def __init__(self, exchange_clients: Dict, telegram_notifier):
        self.exchanges = exchange_clients
        self.telegram = telegram_notifier
        self.emergency_contacts = []
        self.shutdown_callbacks = []
        self.is_killed = False

        # 시그널 핸들러 등록
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """시스템 시그널 처리"""
        logger.critical(f"Received signal {signum}. Initiating emergency shutdown.")
        self.emergency_stop(reason="System signal received")

    async def emergency_stop(self, reason: str = "Manual trigger"):
        """긴급 정지 실행"""

        if self.is_killed:
            return  # 이미 정지됨

        self.is_killed = True
        logger.critical(f"EMERGENCY STOP INITIATED: {reason}")

        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'reason': reason,
            'actions': []
        }

        # 1. 모든 미체결 주문 취소
        for exchange_name, client in self.exchanges.items():
            try:
                open_orders = await client.fetch_open_orders()
                for order in open_orders:
                    await client.cancel_order(order['id'], order['symbol'])
                    results['actions'].append(f"Cancelled order {order['id']}")

                logger.info(f"{exchange_name}: Cancelled {len(open_orders)} orders")

            except Exception as e:
                logger.error(f"Failed to cancel orders on {exchange_name}: {e}")

        # 2. 모든 포지션 즉시 청산
        for exchange_name, client in self.exchanges.items():
            try:
                positions = await self._fetch_positions(client)

                for symbol, quantity in positions.items():
                    if quantity > 0:
                        # 시장가 매도
                        order = await client.create_market_sell_order(symbol, quantity)
                        results['actions'].append(f"Liquidated {quantity} {symbol}")
                        logger.info(f"Liquidated position: {quantity} {symbol}")

            except Exception as e:
                logger.error(f"Failed to liquidate on {exchange_name}: {e}")

        # 3. 거래 봇 프로세스 정지
        for callback in self.shutdown_callbacks:
            try:
                await callback()
            except Exception as e:
                logger.error(f"Shutdown callback failed: {e}")

        # 4. 재시작 방지 플래그
        with open('.kill_switch_active', 'w') as f:
            f.write(json.dumps({
                'activated_at': datetime.utcnow().isoformat(),
                'reason': reason,
                'results': results
            }))

        # 5. 긴급 알림 발송
        alert_message = f"""
        🚨🚨🚨 EMERGENCY STOP ACTIVATED 🚨🚨🚨

        Reason: {reason}
        Time: {results['timestamp']}

        Actions taken:
        - Cancelled all open orders
        - Liquidated all positions
        - Stopped trading bot

        ⚠️ MANUAL INTERVENTION REQUIRED

        Check the system before restarting!
        """

        # Telegram
        await self.telegram.send_emergency_alert(alert_message)

        # SMS/Phone (옵션)
        for contact in self.emergency_contacts:
            await self._send_sms(contact, f"TRADING BOT EMERGENCY STOP: {reason}")

        # 6. 로그 저장
        with open(f"emergency_stop_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(results, f, indent=2)

        logger.critical("Emergency stop completed. System halted.")

        # 프로세스 종료
        sys.exit(1)

    def check_kill_switch_status(self) -> bool:
        """Kill Switch 상태 확인"""
        if os.path.exists('.kill_switch_active'):
            with open('.kill_switch_active', 'r') as f:
                data = json.load(f)
                logger.warning(f"Kill switch is active since {data['activated_at']}")
                return True
        return False

    def reset_kill_switch(self, admin_password: str) -> bool:
        """Kill Switch 리셋 (관리자 권한)"""
        # 비밀번호 확인
        password_hash = hashlib.sha256(admin_password.encode()).hexdigest()

        if password_hash != os.environ.get('ADMIN_PASSWORD_HASH'):
            logger.error("Invalid admin password for kill switch reset")
            return False

        # Kill Switch 해제
        if os.path.exists('.kill_switch_active'):
            # 백업
            os.rename('.kill_switch_active',
                     f'.kill_switch_backup_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}')

            self.is_killed = False
            logger.info("Kill switch reset by admin")
            return True

        return False

    def register_shutdown_callback(self, callback: Callable):
        """종료 콜백 등록"""
        self.shutdown_callbacks.append(callback)

    async def _fetch_positions(self, exchange_client) -> Dict[str, float]:
        """포지션 조회 (거래소별 구현)"""
        balance = await exchange_client.fetch_balance()
        positions = {}

        for currency, info in balance.items():
            if info['total'] > 0 and currency != 'USDT':
                positions[f"{currency}/USDT"] = info['total']

        return positions

    async def _send_sms(self, phone_number: str, message: str):
        """SMS 발송 (Twilio 등 사용)"""
        # 실제 구현 필요
        pass
```

---

## 6. 성능 최적화 전략

### 6.1. WebSocket 연결 관리

```python
import asyncio
from typing import Dict, Callable
from collections import deque

class WebSocketManager:
    """WebSocket 연결 관리 및 재연결"""

    def __init__(self, max_reconnect_attempts: int = 5):
        self.connections = {}  # symbol -> ws_connection
        self.callbacks = {}  # symbol -> callback_function
        self.reconnect_attempts = {}
        self.max_reconnect_attempts = max_reconnect_attempts
        self.message_buffer = deque(maxlen=10000)  # 메시지 버퍼

    async def subscribe(self,
                       exchange_client,
                       symbol: str,
                       channels: List[str],
                       callback: Callable):
        """WebSocket 구독"""

        self.callbacks[symbol] = callback
        self.reconnect_attempts[symbol] = 0

        await self._connect_and_subscribe(exchange_client, symbol, channels)

    async def _connect_and_subscribe(self, exchange_client, symbol: str, channels: List[str]):
        """연결 및 구독 실행"""

        try:
            # WebSocket 연결
            if 'ticker' in channels:
                asyncio.create_task(
                    self._handle_ticker_stream(exchange_client, symbol)
                )

            if 'orderbook' in channels:
                asyncio.create_task(
                    self._handle_orderbook_stream(exchange_client, symbol)
                )

            if 'trades' in channels:
                asyncio.create_task(
                    self._handle_trades_stream(exchange_client, symbol)
                )

            logger.info(f"WebSocket connected for {symbol}: {channels}")
            self.reconnect_attempts[symbol] = 0

        except Exception as e:
            logger.error(f"WebSocket connection failed for {symbol}: {e}")
            await self._handle_disconnect(exchange_client, symbol, channels)

    async def _handle_ticker_stream(self, exchange_client, symbol: str):
        """티커 스트림 처리"""

        try:
            while True:
                ticker = await exchange_client.watch_ticker(symbol)

                # 버퍼에 추가
                self.message_buffer.append({
                    'type': 'ticker',
                    'symbol': symbol,
                    'data': ticker,
                    'timestamp': datetime.utcnow()
                })

                # 콜백 실행
                if symbol in self.callbacks:
                    await self.callbacks[symbol]('ticker', ticker)

        except Exception as e:
            logger.error(f"Ticker stream error for {symbol}: {e}")
            await self._handle_disconnect(exchange_client, symbol, ['ticker'])

    async def _handle_disconnect(self, exchange_client, symbol: str, channels: List[str]):
        """연결 끊김 처리"""

        self.reconnect_attempts[symbol] += 1

        if self.reconnect_attempts[symbol] > self.max_reconnect_attempts:
            logger.critical(f"Max reconnection attempts reached for {symbol}")
            # 긴급 알림
            return

        # 지수 백오프
        wait_time = min(2 ** self.reconnect_attempts[symbol], 60)
        logger.info(f"Reconnecting {symbol} in {wait_time} seconds...")

        await asyncio.sleep(wait_time)
        await self._connect_and_subscribe(exchange_client, symbol, channels)

    def get_buffered_messages(self, symbol: str = None,
                            message_type: str = None,
                            limit: int = 100) -> List[Dict]:
        """버퍼된 메시지 조회"""

        messages = list(self.message_buffer)

        if symbol:
            messages = [m for m in messages if m['symbol'] == symbol]

        if message_type:
            messages = [m for m in messages if m['type'] == message_type]

        return messages[-limit:]

    async def close_all(self):
        """모든 WebSocket 연결 종료"""

        for symbol in self.connections:
            try:
                await self.connections[symbol].close()
            except:
                pass

        self.connections.clear()
        logger.info("All WebSocket connections closed")
```

### 6.2. 데이터베이스 최적화

```python
import sqlite3
from contextlib import contextmanager
from typing import List, Dict
import pandas as pd

class OptimizedDatabase:
    """최적화된 데이터베이스 관리"""

    def __init__(self, db_path: str = 'trading.db'):
        self.db_path = db_path
        self.connection_pool = []
        self.max_connections = 5

        # 초기 설정
        self._initialize_db()

    def _initialize_db(self):
        """DB 초기화 및 최적화"""

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # WAL 모드 활성화 (읽기/쓰기 동시성 향상)
            cursor.execute("PRAGMA journal_mode=WAL")

            # 캐시 크기 증가
            cursor.execute("PRAGMA cache_size=10000")

            # 동기화 모드 (성능 vs 안정성)
            cursor.execute("PRAGMA synchronous=NORMAL")

            # 인덱스 생성
            self._create_indexes(cursor)

            # 파티션 테이블 생성
            self._create_partitioned_tables(cursor)

            conn.commit()

    def _create_indexes(self, cursor):
        """인덱스 생성"""

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data(symbol, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_ai_decisions_model_time ON ai_decisions(model, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_system_logs_level_time ON system_logs(level, timestamp DESC)"
        ]

        for idx in indexes:
            cursor.execute(idx)

    def _create_partitioned_tables(self, cursor):
        """파티션 테이블 생성 (월별)"""

        # 현재 월 테이블
        current_month = datetime.utcnow().strftime('%Y%m')

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS market_data_{current_month} (
                CHECK (strftime('%Y%m', timestamp) = '{current_month}')
            ) INHERITS (market_data)
        """)

    @contextmanager
    def _get_connection(self):
        """커넥션 풀에서 연결 획득"""

        if self.connection_pool:
            conn = self.connection_pool.pop()
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # 딕셔너리 형태 결과

        try:
            yield conn
        finally:
            if len(self.connection_pool) < self.max_connections:
                self.connection_pool.append(conn)
            else:
                conn.close()

    def bulk_insert_market_data(self, data: List[Dict]):
        """대량 시장 데이터 삽입"""

        with self._get_connection() as conn:
            df = pd.DataFrame(data)

            # 월별 파티션 결정
            df['partition'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y%m')

            for partition, group in df.groupby('partition'):
                table_name = f"market_data_{partition}"

                # 테이블 존재 확인
                cursor = conn.cursor()
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} AS
                    SELECT * FROM market_data WHERE 0
                """)

                # 대량 삽입
                group.to_sql(table_name, conn, if_exists='append', index=False)

            conn.commit()
            logger.info(f"Bulk inserted {len(data)} market data records")

    def query_with_cache(self, query: str, params: tuple = ()) -> List[Dict]:
        """캐시를 활용한 쿼리"""

        # 간단한 메모리 캐시 (실제로는 Redis 등 사용)
        cache_key = hashlib.md5(f"{query}{params}".encode()).hexdigest()

        if hasattr(self, '_cache') and cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if datetime.utcnow() - cache_entry['timestamp'] < timedelta(seconds=60):
                return cache_entry['data']

        # DB 쿼리
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]

        # 캐시 저장
        if not hasattr(self, '_cache'):
            self._cache = {}

        self._cache[cache_key] = {
            'data': results,
            'timestamp': datetime.utcnow()
        }

        return results

    def cleanup_old_data(self, days_to_keep: int = 90):
        """오래된 데이터 정리"""

        cutoff_date = (datetime.utcnow() - timedelta(days=days_to_keep)).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 오래된 파티션 테이블 삭제
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name LIKE 'market_data_%'
            """)

            tables = cursor.fetchall()
            for table in tables:
                table_month = table['name'].split('_')[-1]
                if table_month < (datetime.utcnow() - timedelta(days=days_to_keep)).strftime('%Y%m'):
                    cursor.execute(f"DROP TABLE {table['name']}")
                    logger.info(f"Dropped old partition: {table['name']}")

            # 다른 테이블 정리
            tables_to_clean = ['trades', 'ai_decisions', 'system_logs']
            for table in tables_to_clean:
                cursor.execute(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff_date,))

            # VACUUM으로 공간 회수
            cursor.execute("VACUUM")

            conn.commit()
            logger.info(f"Cleaned up data older than {days_to_keep} days")
```

---

## 7. 백테스팅 현실성 개선

### 7.1. 현실적 백테스팅 엔진

```python
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class BacktestConfig:
    """백테스트 설정"""
    initial_balance: float = 10000
    commission_rate: float = 0.001  # 0.1%
    slippage_model: str = 'linear'  # 'linear', 'square_root', 'logarithmic'
    slippage_factor: float = 0.001  # 0.1%
    min_spread_pct: float = 0.0001  # 0.01%
    latency_ms: int = 50  # 네트워크 지연
    api_failure_rate: float = 0.001  # 0.1% API 실패율

class RealisticBacktester:
    """현실적 백테스팅 엔진"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.balance = config.initial_balance
        self.positions = {}
        self.trades = []
        self.order_book_depth = {}

    def simulate_market_order(self,
                            symbol: str,
                            side: str,
                            quantity: float,
                            orderbook: Dict,
                            timestamp: datetime) -> Dict:
        """현실적 시장가 주문 시뮬레이션"""

        # 1. 호가창 깊이 확인
        if side == 'BUY':
            orders = orderbook['asks']
        else:
            orders = orderbook['bids']

        if not orders:
            return {'status': 'rejected', 'reason': 'No liquidity'}

        # 2. 슬리피지 계산
        filled_quantity = 0
        total_cost = 0
        fills = []

        for price, volume in orders:
            if filled_quantity >= quantity:
                break

            fill_qty = min(volume, quantity - filled_quantity)

            # 슬리피지 적용
            slippage = self._calculate_slippage(fill_qty, volume)
            adjusted_price = price * (1 + slippage if side == 'BUY' else 1 - slippage)

            fills.append({
                'price': adjusted_price,
                'quantity': fill_qty
            })

            total_cost += adjusted_price * fill_qty
            filled_quantity += fill_qty

        if filled_quantity < quantity:
            # 부분 체결
            logger.warning(f"Partial fill: {filled_quantity}/{quantity}")

        # 3. 평균 체결가
        avg_price = total_cost / filled_quantity if filled_quantity > 0 else 0

        # 4. 수수료 계산
        commission = total_cost * self.config.commission_rate

        # 5. 지연 시뮬레이션
        execution_timestamp = timestamp + timedelta(milliseconds=self.config.latency_ms)

        # 6. API 실패 시뮬레이션
        if np.random.random() < self.config.api_failure_rate:
            return {'status': 'failed', 'reason': 'API error'}

        # 7. 잔고 업데이트
        if side == 'BUY':
            if self.balance < total_cost + commission:
                return {'status': 'rejected', 'reason': 'Insufficient balance'}

            self.balance -= (total_cost + commission)

            if symbol not in self.positions:
                self.positions[symbol] = 0
            self.positions[symbol] += filled_quantity

        else:  # SELL
            if symbol not in self.positions or self.positions[symbol] < filled_quantity:
                return {'status': 'rejected', 'reason': 'Insufficient position'}

            self.positions[symbol] -= filled_quantity
            self.balance += (total_cost - commission)

        # 8. 거래 기록
        trade = {
            'timestamp': execution_timestamp,
            'symbol': symbol,
            'side': side,
            'quantity': filled_quantity,
            'avg_price': avg_price,
            'commission': commission,
            'slippage_pct': (avg_price - orders[0][0]) / orders[0][0] * 100,
            'fills': fills
        }

        self.trades.append(trade)

        return {
            'status': 'filled',
            'trade': trade
        }

    def _calculate_slippage(self, order_size: float, available_liquidity: float) -> float:
        """슬리피지 모델링"""

        impact_ratio = order_size / available_liquidity if available_liquidity > 0 else 1

        if self.config.slippage_model == 'linear':
            # 선형 모델
            slippage = self.config.slippage_factor * impact_ratio

        elif self.config.slippage_model == 'square_root':
            # 제곱근 모델 (큰 주문에 더 관대)
            slippage = self.config.slippage_factor * np.sqrt(impact_ratio)

        elif self.config.slippage_model == 'logarithmic':
            # 로그 모델
            slippage = self.config.slippage_factor * np.log(1 + impact_ratio)

        else:
            slippage = 0

        # 최소 스프레드 보장
        slippage = max(slippage, self.config.min_spread_pct)

        # 랜덤 요소 추가 (현실성)
        slippage *= np.random.uniform(0.8, 1.2)

        return slippage

    def simulate_limit_order(self,
                           symbol: str,
                           side: str,
                           quantity: float,
                           limit_price: float,
                           market_data_stream: List[Dict]) -> Dict:
        """지정가 주문 시뮬레이션"""

        order = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'limit_price': limit_price,
            'status': 'pending',
            'filled_quantity': 0
        }

        # 시장 데이터 스트림에서 체결 시뮬레이션
        for tick in market_data_stream:
            if side == 'BUY' and tick['ask'] <= limit_price:
                # 매수 체결
                order['status'] = 'filled'
                order['fill_price'] = tick['ask']
                order['fill_time'] = tick['timestamp']
                break

            elif side == 'SELL' and tick['bid'] >= limit_price:
                # 매도 체결
                order['status'] = 'filled'
                order['fill_price'] = tick['bid']
                order['fill_time'] = tick['timestamp']
                break

        return order

    def calculate_metrics(self) -> Dict:
        """백테스트 성과 지표"""

        if not self.trades:
            return {'error': 'No trades executed'}

        df_trades = pd.DataFrame(self.trades)

        # PnL 계산
        buy_trades = df_trades[df_trades['side'] == 'BUY']
        sell_trades = df_trades[df_trades['side'] == 'SELL']

        # 매매 쌍 매칭 (FIFO)
        pairs = []
        for _, sell in sell_trades.iterrows():
            matching_buys = buy_trades[
                (buy_trades['symbol'] == sell['symbol']) &
                (buy_trades['timestamp'] < sell['timestamp'])
            ]

            if not matching_buys.empty:
                buy = matching_buys.iloc[0]
                pnl = (sell['avg_price'] - buy['avg_price']) * sell['quantity']
                pnl -= (buy['commission'] + sell['commission'])
                pairs.append(pnl)

        # 지표 계산
        returns = np.array(pairs)
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        metrics = {
            'total_trades': len(self.trades),
            'win_rate': len(positive_returns) / len(returns) * 100 if len(returns) > 0 else 0,
            'avg_win': np.mean(positive_returns) if len(positive_returns) > 0 else 0,
            'avg_loss': np.mean(negative_returns) if len(negative_returns) > 0 else 0,
            'profit_factor': abs(np.sum(positive_returns) / np.sum(negative_returns)) if np.sum(negative_returns) != 0 else 0,
            'total_pnl': np.sum(returns),
            'total_commission': df_trades['commission'].sum(),
            'avg_slippage': df_trades['slippage_pct'].mean(),
            'max_slippage': df_trades['slippage_pct'].max(),
            'final_balance': self.balance,
            'roi': (self.balance - self.config.initial_balance) / self.config.initial_balance * 100
        }

        return metrics
```

---

## 문서 요약

이 보완 문서는 PRD와 구축 계획서에서 누락된 실전 트레이딩의 핵심 기능들을 상세히 다룹니다:

### 추가된 핵심 기능

1. **거래소 제약사항 검증**: 최소 주문 금액, 틱 사이즈, 수량 정밀도 관리
2. **부분 체결 처리**: 실전에서 빈번히 발생하는 부분 체결 상황 대응
3. **슬리피지 예측**: 호가창 분석을 통한 실제 체결가 예측
4. **고급 포지션 관리**: 평균 단가 추적, 미실현 손익 계산, 포트폴리오 리밸런싱
5. **Paper Trading 엔진**: 실전과 동일한 환경의 모의 거래 시스템
6. **보안 강화**: API 키 암호화, IP 화이트리스트, 강화된 Kill Switch
7. **성능 최적화**: WebSocket 재연결, DB 파티셔닝, 캐싱 전략
8. **현실적 백테스팅**: 슬리피지, 수수료, API 실패율을 반영한 시뮬레이션

### 구현 우선순위

**즉시 구현 필요 (Phase 1-2)**:
- 거래소 제약사항 검증
- 부분 체결 처리
- 기본 Kill Switch

**중요 기능 (Phase 3-4)**:
- Paper Trading 엔진
- 포지션 추적 시스템
- WebSocket 관리

**고도화 (Phase 5-6)**:
- 슬리피지 예측
- 포트폴리오 리밸런싱
- 현실적 백테스팅

이러한 기능들을 구현함으로써 실전 거래에서 발생할 수 있는 다양한 상황에 대응 가능한 견고한 시스템을 구축할 수 있습니다.