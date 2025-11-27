# 리스크 관리 전략 문서

**작성일:** 2025-11-22
**목적:** AI 가상화폐 자동매매 시스템의 종합적인 리스크 관리 체계 구축

---

## 목차

1. [리스크 분류 및 우선순위](#1-리스크-분류-및-우선순위)
2. [자금 관리 (Money Management)](#2-자금-관리-money-management)
3. [포지션 리스크 관리](#3-포지션-리스크-관리)
4. [시스템 리스크 관리](#4-시스템-리스크-관리)
5. [AI 모델 리스크 관리](#5-ai-모델-리스크-관리)
6. [운영 리스크 관리](#6-운영-리스크-관리)
7. [규제 및 컴플라이언스](#7-규제-및-컴플라이언스)

---

## 1. 리스크 분류 및 우선순위

### 1.1. 리스크 매트릭스

| 리스크 유형 | 발생 가능성 | 영향도 | 우선순위 | 대응 전략 |
|------------|------------|--------|---------|-----------|
| **시장 리스크** | 높음 | 높음 | 1 | Stop-loss, 포지션 제한 |
| **유동성 리스크** | 중간 | 높음 | 2 | 주문 분할, 슬리피지 관리 |
| **기술적 장애** | 중간 | 높음 | 2 | 이중화, 백업 시스템 |
| **AI 오판단** | 중간 | 중간 | 3 | 검증 레이어, 신뢰도 임계값 |
| **거래소 리스크** | 낮음 | 높음 | 3 | 다중 거래소, 자금 분산 |
| **규제 리스크** | 낮음 | 중간 | 4 | 컴플라이언스 모니터링 |
| **사이버 보안** | 낮음 | 높음 | 4 | 암호화, 접근 제어 |

### 1.2. 리스크 한도 설정

```python
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class RiskLimits:
    """전체 리스크 한도 설정"""

    # 자금 관리
    max_position_size_pct: float = 0.05  # 총 자산의 5%
    max_total_exposure_pct: float = 0.30  # 총 자산의 30%
    max_single_loss_pct: float = 0.02  # 단일 거래 최대 손실 2%

    # 손실 제한
    daily_loss_limit_pct: float = 0.05  # 일일 최대 손실 5%
    weekly_loss_limit_pct: float = 0.10  # 주간 최대 손실 10%
    max_drawdown_pct: float = 0.15  # 최대 낙폭 15%

    # 거래 제한
    max_daily_trades: int = 20  # 일일 최대 거래 횟수
    max_concurrent_positions: int = 5  # 동시 보유 포지션 수
    min_time_between_trades: int = 60  # 거래 간 최소 간격(초)

    # AI 제한
    min_ai_confidence: float = 0.70  # 최소 AI 신뢰도
    max_ai_api_cost_daily: float = 100  # 일일 AI API 비용 한도($)

    # 시장 조건
    max_volatility_threshold: float = 0.10  # 변동성 10% 초과 시 거래 중단
    min_liquidity_usd: float = 100000  # 최소 유동성 $100,000

class RiskManager:
    """통합 리스크 관리자"""

    def __init__(self, limits: RiskLimits = None):
        self.limits = limits or RiskLimits()
        self.current_exposure = {}
        self.daily_stats = {}
        self.alert_callbacks = []

    def check_all_limits(self, trade_request: Dict) -> Dict:
        """모든 리스크 한도 검증"""

        checks = {
            'position_size': self._check_position_size(trade_request),
            'total_exposure': self._check_total_exposure(trade_request),
            'daily_loss': self._check_daily_loss(),
            'drawdown': self._check_drawdown(),
            'trade_frequency': self._check_trade_frequency(),
            'ai_confidence': self._check_ai_confidence(trade_request),
            'market_conditions': self._check_market_conditions(trade_request)
        }

        all_passed = all(check['passed'] for check in checks.values())

        return {
            'approved': all_passed,
            'checks': checks,
            'risk_score': self._calculate_risk_score(checks)
        }

    def _calculate_risk_score(self, checks: Dict) -> float:
        """리스크 점수 계산 (0-1, 낮을수록 안전)"""

        weights = {
            'position_size': 0.20,
            'total_exposure': 0.20,
            'daily_loss': 0.15,
            'drawdown': 0.15,
            'trade_frequency': 0.10,
            'ai_confidence': 0.10,
            'market_conditions': 0.10
        }

        score = sum(
            weights[key] * (0 if check['passed'] else 1)
            for key, check in checks.items()
        )

        return score
```

---

## 2. 자금 관리 (Money Management)

### 2.1. Kelly Criterion 기반 포지션 크기 결정

```python
import numpy as np
from typing import Dict, List

class KellyPositionSizer:
    """Kelly Criterion을 활용한 최적 포지션 크기 계산"""

    def __init__(self, kelly_fraction: float = 0.25):
        """
        kelly_fraction: Full Kelly의 비율 (0.25 = Quarter Kelly, 더 보수적)
        """
        self.kelly_fraction = kelly_fraction
        self.historical_trades = []

    def calculate_optimal_position_size(self,
                                       win_probability: float,
                                       avg_win_loss_ratio: float,
                                       current_capital: float,
                                       max_position_pct: float = 0.05) -> float:
        """
        Kelly Criterion: f = (p*b - q) / b
        where:
        f = 베팅 비율
        p = 승률
        b = 평균 수익/손실 비율
        q = 패배 확률 (1-p)
        """

        if win_probability <= 0 or win_probability >= 1:
            return 0

        q = 1 - win_probability
        b = avg_win_loss_ratio

        # Kelly 비율 계산
        kelly_pct = (win_probability * b - q) / b

        # 음수면 베팅하지 않음
        if kelly_pct <= 0:
            return 0

        # Kelly fraction 적용 (보수적 접근)
        adjusted_kelly = kelly_pct * self.kelly_fraction

        # 최대 포지션 제한
        final_position_pct = min(adjusted_kelly, max_position_pct)

        # 금액 계산
        position_size = current_capital * final_position_pct

        return position_size

    def update_statistics(self, trade_result: Dict):
        """거래 결과로 통계 업데이트"""

        self.historical_trades.append(trade_result)

        # 최근 100개 거래만 유지
        if len(self.historical_trades) > 100:
            self.historical_trades = self.historical_trades[-100:]

    def get_current_statistics(self) -> Dict:
        """현재 거래 통계"""

        if not self.historical_trades:
            return {
                'win_rate': 0.5,  # 기본값
                'avg_win': 0,
                'avg_loss': 0,
                'win_loss_ratio': 1.0
            }

        wins = [t for t in self.historical_trades if t['pnl'] > 0]
        losses = [t for t in self.historical_trades if t['pnl'] < 0]

        win_rate = len(wins) / len(self.historical_trades)
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t['pnl'] for t in losses])) if losses else 1

        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': avg_win / avg_loss if avg_loss > 0 else 0
        }

    def calculate_position_with_var(self,
                                   confidence_level: float = 0.95,
                                   lookback_days: int = 30) -> float:
        """VaR(Value at Risk) 기반 포지션 크기 조정"""

        if len(self.historical_trades) < lookback_days:
            return 0

        # 일별 수익률 계산
        returns = [t['return_pct'] for t in self.historical_trades[-lookback_days:]]

        # VaR 계산 (파라메트릭 방법)
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Z-score (95% 신뢰구간 = 1.645)
        z_score = 1.645 if confidence_level == 0.95 else 2.33

        var = mean_return - (z_score * std_return)

        # VaR 기반 포지션 조정
        # VaR가 클수록 (손실 위험이 클수록) 포지션 감소
        if var < -0.05:  # 5% 이상 손실 위험
            position_multiplier = 0.5
        elif var < -0.03:  # 3% 이상 손실 위험
            position_multiplier = 0.75
        else:
            position_multiplier = 1.0

        return position_multiplier
```

### 2.2. 자금 배분 전략

```python
class CapitalAllocationStrategy:
    """자금 배분 전략"""

    def __init__(self, total_capital: float):
        self.total_capital = total_capital
        self.allocations = {}

    def core_satellite_allocation(self) -> Dict:
        """Core-Satellite 전략"""

        return {
            'core': {
                'allocation_pct': 0.70,  # 70% 안정적 전략
                'amount': self.total_capital * 0.70,
                'strategy': 'conservative',
                'coins': ['BTC', 'ETH'],
                'max_leverage': 1.0
            },
            'satellite': {
                'allocation_pct': 0.20,  # 20% 공격적 전략
                'amount': self.total_capital * 0.20,
                'strategy': 'aggressive',
                'coins': ['ALT_COINS'],
                'max_leverage': 2.0
            },
            'reserve': {
                'allocation_pct': 0.10,  # 10% 현금 보유
                'amount': self.total_capital * 0.10,
                'purpose': 'emergency_liquidity'
            }
        }

    def risk_parity_allocation(self, asset_volatilities: Dict[str, float]) -> Dict:
        """리스크 패리티 전략 (동일 리스크 기여도)"""

        # 변동성의 역수로 가중치 계산
        inv_vols = {asset: 1/vol for asset, vol in asset_volatilities.items()}
        total_inv_vol = sum(inv_vols.values())

        allocations = {}
        for asset, inv_vol in inv_vols.items():
            weight = inv_vol / total_inv_vol
            allocations[asset] = {
                'weight': weight,
                'amount': self.total_capital * weight,
                'volatility': asset_volatilities[asset]
            }

        return allocations

    def dynamic_allocation(self, market_regime: str) -> Dict:
        """시장 상황에 따른 동적 배분"""

        regimes = {
            'bull': {
                'risk_assets': 0.80,
                'stable_assets': 0.10,
                'cash': 0.10
            },
            'bear': {
                'risk_assets': 0.20,
                'stable_assets': 0.30,
                'cash': 0.50
            },
            'sideways': {
                'risk_assets': 0.40,
                'stable_assets': 0.40,
                'cash': 0.20
            }
        }

        allocation = regimes.get(market_regime, regimes['sideways'])

        return {
            'risk_assets': self.total_capital * allocation['risk_assets'],
            'stable_assets': self.total_capital * allocation['stable_assets'],
            'cash': self.total_capital * allocation['cash'],
            'regime': market_regime
        }
```

---

## 3. 포지션 리스크 관리

### 3.1. Stop-Loss 전략

```python
from typing import Dict, Optional, List
from datetime import datetime, timedelta

class StopLossManager:
    """다양한 Stop-Loss 전략 관리"""

    def __init__(self):
        self.active_stops = {}  # position_id -> stop_info

    def set_fixed_stop_loss(self,
                           position_id: str,
                           entry_price: float,
                           stop_loss_pct: float = 0.07) -> Dict:
        """고정 Stop-Loss 설정"""

        stop_price = entry_price * (1 - stop_loss_pct)

        self.active_stops[position_id] = {
            'type': 'fixed',
            'entry_price': entry_price,
            'stop_price': stop_price,
            'stop_loss_pct': stop_loss_pct,
            'created_at': datetime.utcnow()
        }

        return self.active_stops[position_id]

    def set_trailing_stop_loss(self,
                              position_id: str,
                              current_price: float,
                              trailing_pct: float = 0.05) -> Dict:
        """Trailing Stop-Loss 설정"""

        if position_id not in self.active_stops:
            # 초기 설정
            self.active_stops[position_id] = {
                'type': 'trailing',
                'highest_price': current_price,
                'stop_price': current_price * (1 - trailing_pct),
                'trailing_pct': trailing_pct,
                'created_at': datetime.utcnow()
            }
        else:
            # 기존 trailing stop 업데이트
            stop_info = self.active_stops[position_id]

            if current_price > stop_info['highest_price']:
                stop_info['highest_price'] = current_price
                stop_info['stop_price'] = current_price * (1 - trailing_pct)
                stop_info['updated_at'] = datetime.utcnow()

        return self.active_stops[position_id]

    def set_atr_based_stop(self,
                          position_id: str,
                          entry_price: float,
                          atr_value: float,
                          atr_multiplier: float = 2.0) -> Dict:
        """ATR 기반 동적 Stop-Loss"""

        stop_distance = atr_value * atr_multiplier
        stop_price = entry_price - stop_distance

        self.active_stops[position_id] = {
            'type': 'atr_based',
            'entry_price': entry_price,
            'stop_price': stop_price,
            'atr_value': atr_value,
            'atr_multiplier': atr_multiplier,
            'created_at': datetime.utcnow()
        }

        return self.active_stops[position_id]

    def set_time_based_stop(self,
                           position_id: str,
                           entry_price: float,
                           time_limit_hours: int = 24,
                           decay_rate: float = 0.01) -> Dict:
        """시간 기반 Stop-Loss (시간이 지날수록 타이트해짐)"""

        self.active_stops[position_id] = {
            'type': 'time_based',
            'entry_price': entry_price,
            'initial_stop_pct': 0.10,  # 초기 10%
            'time_limit_hours': time_limit_hours,
            'decay_rate': decay_rate,  # 시간당 축소율
            'created_at': datetime.utcnow()
        }

        return self.active_stops[position_id]

    def check_stop_loss(self, position_id: str, current_price: float) -> Dict:
        """Stop-Loss 체크"""

        if position_id not in self.active_stops:
            return {'triggered': False, 'reason': 'No stop-loss set'}

        stop_info = self.active_stops[position_id]

        if stop_info['type'] == 'fixed':
            if current_price <= stop_info['stop_price']:
                return {
                    'triggered': True,
                    'reason': 'Fixed stop-loss triggered',
                    'stop_price': stop_info['stop_price'],
                    'loss_pct': (current_price - stop_info['entry_price']) / stop_info['entry_price'] * 100
                }

        elif stop_info['type'] == 'trailing':
            if current_price <= stop_info['stop_price']:
                return {
                    'triggered': True,
                    'reason': 'Trailing stop-loss triggered',
                    'stop_price': stop_info['stop_price'],
                    'max_profit_pct': (stop_info['highest_price'] - stop_info.get('entry_price', stop_info['highest_price'])) / stop_info.get('entry_price', stop_info['highest_price']) * 100
                }

        elif stop_info['type'] == 'time_based':
            # 시간 경과 계산
            hours_elapsed = (datetime.utcnow() - stop_info['created_at']).total_seconds() / 3600
            current_stop_pct = max(
                stop_info['initial_stop_pct'] - (hours_elapsed * stop_info['decay_rate']),
                0.02  # 최소 2%
            )
            stop_price = stop_info['entry_price'] * (1 - current_stop_pct)

            if current_price <= stop_price:
                return {
                    'triggered': True,
                    'reason': 'Time-based stop-loss triggered',
                    'stop_price': stop_price,
                    'hours_held': hours_elapsed
                }

        return {'triggered': False}

    def get_stop_loss_statistics(self) -> Dict:
        """Stop-Loss 통계"""

        stats = {
            'total_stops': len(self.active_stops),
            'by_type': {},
            'average_stop_distance': []
        }

        for position_id, stop_info in self.active_stops.items():
            stop_type = stop_info['type']

            if stop_type not in stats['by_type']:
                stats['by_type'][stop_type] = 0
            stats['by_type'][stop_type] += 1

            if 'entry_price' in stop_info and 'stop_price' in stop_info:
                distance = abs(stop_info['stop_price'] - stop_info['entry_price']) / stop_info['entry_price'] * 100
                stats['average_stop_distance'].append(distance)

        if stats['average_stop_distance']:
            stats['average_stop_distance'] = np.mean(stats['average_stop_distance'])
        else:
            stats['average_stop_distance'] = 0

        return stats
```

### 3.2. 포지션 상관관계 관리

```python
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

class CorrelationRiskManager:
    """포지션 간 상관관계 리스크 관리"""

    def __init__(self, max_correlation: float = 0.7):
        self.max_correlation = max_correlation
        self.price_history = {}
        self.correlation_matrix = None

    def update_price_history(self, symbol: str, prices: List[float]):
        """가격 히스토리 업데이트"""
        self.price_history[symbol] = prices

        # 상관관계 매트릭스 재계산
        if len(self.price_history) > 1:
            self._calculate_correlation_matrix()

    def _calculate_correlation_matrix(self):
        """상관관계 매트릭스 계산"""

        # DataFrame 생성
        df = pd.DataFrame(self.price_history)

        # 수익률 계산
        returns = df.pct_change().dropna()

        # 상관관계 매트릭스
        self.correlation_matrix = returns.corr()

    def check_correlation_risk(self, new_position: str, existing_positions: List[str]) -> Dict:
        """새 포지션의 상관관계 리스크 체크"""

        if self.correlation_matrix is None:
            return {'allowed': True, 'reason': 'Insufficient data'}

        high_correlations = []

        for existing in existing_positions:
            if existing in self.correlation_matrix and new_position in self.correlation_matrix:
                correlation = self.correlation_matrix.loc[new_position, existing]

                if abs(correlation) > self.max_correlation:
                    high_correlations.append({
                        'symbol': existing,
                        'correlation': correlation
                    })

        if high_correlations:
            return {
                'allowed': False,
                'reason': 'High correlation with existing positions',
                'correlations': high_correlations
            }

        return {'allowed': True}

    def calculate_portfolio_var(self, positions: Dict[str, float], confidence_level: float = 0.95) -> float:
        """포트폴리오 VaR 계산"""

        if not positions or self.correlation_matrix is None:
            return 0

        # 포지션 가중치
        total_value = sum(positions.values())
        weights = np.array([positions.get(s, 0) / total_value for s in self.correlation_matrix.columns])

        # 공분산 매트릭스
        returns = pd.DataFrame(self.price_history).pct_change().dropna()
        cov_matrix = returns.cov()

        # 포트폴리오 분산
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)

        # VaR 계산
        z_score = 1.645 if confidence_level == 0.95 else 2.33
        portfolio_var = portfolio_std * z_score * np.sqrt(1)  # 1일 VaR

        return portfolio_var

    def suggest_diversification(self, current_positions: List[str]) -> List[str]:
        """다각화 제안"""

        if self.correlation_matrix is None:
            return []

        suggestions = []

        # 현재 포지션과 낮은 상관관계를 가진 자산 찾기
        for symbol in self.correlation_matrix.columns:
            if symbol not in current_positions:
                avg_correlation = np.mean([
                    abs(self.correlation_matrix.loc[symbol, pos])
                    for pos in current_positions
                    if pos in self.correlation_matrix
                ])

                if avg_correlation < 0.3:  # 낮은 상관관계
                    suggestions.append({
                        'symbol': symbol,
                        'avg_correlation': avg_correlation
                    })

        # 상관관계가 낮은 순으로 정렬
        suggestions.sort(key=lambda x: x['avg_correlation'])

        return suggestions[:5]  # 상위 5개 제안
```

---

## 4. 시스템 리스크 관리

### 4.1. 장애 복구 시스템

```python
import asyncio
from typing import Dict, List, Optional
from datetime import datetime

class DisasterRecoverySystem:
    """재해 복구 시스템"""

    def __init__(self):
        self.backup_systems = []
        self.health_checks = {}
        self.failover_state = 'primary'
        self.recovery_procedures = {}

    async def setup_redundancy(self):
        """이중화 시스템 설정"""

        # Primary 시스템
        self.primary_system = {
            'name': 'primary',
            'endpoint': 'https://primary.trading-bot.com',
            'database': 'primary_db',
            'status': 'active'
        }

        # Backup 시스템
        self.backup_system = {
            'name': 'backup',
            'endpoint': 'https://backup.trading-bot.com',
            'database': 'backup_db',
            'status': 'standby'
        }

        # 데이터 동기화 설정
        asyncio.create_task(self.sync_data_continuous())

    async def sync_data_continuous(self):
        """지속적 데이터 동기화"""

        while True:
            try:
                # 주요 데이터 동기화
                await self._sync_positions()
                await self._sync_orders()
                await self._sync_configurations()

                logger.info("Data sync completed successfully")

            except Exception as e:
                logger.error(f"Data sync failed: {e}")
                await self._alert_sync_failure(e)

            await asyncio.sleep(60)  # 1분마다 동기화

    async def _sync_positions(self):
        """포지션 데이터 동기화"""

        # Primary에서 데이터 읽기
        primary_positions = await self._fetch_from_primary('positions')

        # Backup에 쓰기
        await self._write_to_backup('positions', primary_positions)

    async def check_system_health(self) -> Dict:
        """시스템 건강 상태 확인"""

        checks = {
            'database': await self._check_database(),
            'api_connections': await self._check_api_connections(),
            'disk_space': await self._check_disk_space(),
            'memory_usage': await self._check_memory_usage(),
            'network_latency': await self._check_network_latency()
        }

        overall_health = 'healthy'
        for component, status in checks.items():
            if status['status'] == 'critical':
                overall_health = 'critical'
                break
            elif status['status'] == 'warning' and overall_health == 'healthy':
                overall_health = 'warning'

        return {
            'overall': overall_health,
            'components': checks,
            'timestamp': datetime.utcnow()
        }

    async def execute_failover(self, reason: str):
        """Failover 실행"""

        logger.critical(f"Executing failover: {reason}")

        try:
            # 1. Primary 시스템 정지
            await self._stop_primary_system()

            # 2. Backup 시스템 활성화
            await self._activate_backup_system()

            # 3. DNS/로드밸런서 전환
            await self._switch_traffic_to_backup()

            # 4. 알림 발송
            await self._notify_failover(reason)

            self.failover_state = 'backup'
            logger.info("Failover completed successfully")

            return {'status': 'success', 'active_system': 'backup'}

        except Exception as e:
            logger.error(f"Failover failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    async def execute_failback(self):
        """Primary 시스템으로 복구"""

        if self.failover_state != 'backup':
            return {'status': 'not_in_failover'}

        try:
            # 1. Primary 시스템 상태 확인
            primary_health = await self._check_primary_health()

            if primary_health != 'healthy':
                return {'status': 'primary_not_ready'}

            # 2. 데이터 역동기화
            await self._sync_backup_to_primary()

            # 3. Primary 시스템 재활성화
            await self._reactivate_primary_system()

            # 4. 트래픽 전환
            await self._switch_traffic_to_primary()

            self.failover_state = 'primary'
            logger.info("Failback completed successfully")

            return {'status': 'success', 'active_system': 'primary'}

        except Exception as e:
            logger.error(f"Failback failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def create_recovery_plan(self) -> Dict:
        """복구 계획 생성"""

        return {
            'RTO': '5 minutes',  # Recovery Time Objective
            'RPO': '1 minute',   # Recovery Point Objective
            'procedures': {
                'database_failure': self._database_recovery_procedure(),
                'api_failure': self._api_recovery_procedure(),
                'network_failure': self._network_recovery_procedure(),
                'complete_failure': self._complete_failure_procedure()
            },
            'contact_list': self._emergency_contacts(),
            'backup_locations': self._backup_locations()
        }

    def _database_recovery_procedure(self) -> List[str]:
        """데이터베이스 복구 절차"""

        return [
            "1. 데이터베이스 상태 확인 (SHOW STATUS)",
            "2. 손상된 테이블 확인 (CHECK TABLE)",
            "3. 백업에서 복구 (RESTORE DATABASE)",
            "4. 트랜잭션 로그 적용 (APPLY LOG)",
            "5. 데이터 무결성 검증 (VERIFY CHECKSUM)",
            "6. 서비스 재시작"
        ]
```

### 4.2. 모니터링 및 알림 시스템

```python
from enum import Enum
from typing import Dict, List, Callable
import asyncio

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MonitoringSystem:
    """종합 모니터링 시스템"""

    def __init__(self):
        self.metrics = {}
        self.alert_rules = []
        self.alert_channels = []
        self.alert_history = []

    def add_metric(self, name: str, value: float, timestamp: datetime = None):
        """메트릭 추가"""

        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append({
            'value': value,
            'timestamp': timestamp or datetime.utcnow()
        })

        # 알림 규칙 체크
        self._check_alert_rules(name, value)

    def add_alert_rule(self,
                      metric_name: str,
                      condition: str,
                      threshold: float,
                      level: AlertLevel,
                      message_template: str):
        """알림 규칙 추가"""

        self.alert_rules.append({
            'metric': metric_name,
            'condition': condition,  # '>', '<', '==', etc.
            'threshold': threshold,
            'level': level,
            'message_template': message_template
        })

    def _check_alert_rules(self, metric_name: str, value: float):
        """알림 규칙 체크"""

        for rule in self.alert_rules:
            if rule['metric'] != metric_name:
                continue

            triggered = False
            if rule['condition'] == '>':
                triggered = value > rule['threshold']
            elif rule['condition'] == '<':
                triggered = value < rule['threshold']
            elif rule['condition'] == '==':
                triggered = value == rule['threshold']

            if triggered:
                self._trigger_alert(rule, value)

    def _trigger_alert(self, rule: Dict, value: float):
        """알림 발송"""

        alert = {
            'level': rule['level'],
            'metric': rule['metric'],
            'value': value,
            'threshold': rule['threshold'],
            'message': rule['message_template'].format(
                metric=rule['metric'],
                value=value,
                threshold=rule['threshold']
            ),
            'timestamp': datetime.utcnow()
        }

        self.alert_history.append(alert)

        # 모든 채널로 알림 발송
        for channel in self.alert_channels:
            asyncio.create_task(channel.send(alert))

    def setup_default_monitors(self):
        """기본 모니터링 설정"""

        # API 응답 시간
        self.add_alert_rule(
            'api_latency_ms',
            '>',
            1000,
            AlertLevel.WARNING,
            "API latency high: {value}ms (threshold: {threshold}ms)"
        )

        # 메모리 사용률
        self.add_alert_rule(
            'memory_usage_pct',
            '>',
            80,
            AlertLevel.ERROR,
            "Memory usage critical: {value}% (threshold: {threshold}%)"
        )

        # 일일 손실
        self.add_alert_rule(
            'daily_loss_pct',
            '<',
            -5,
            AlertLevel.CRITICAL,
            "Daily loss exceeded: {value}% (limit: {threshold}%)"
        )

        # AI API 비용
        self.add_alert_rule(
            'ai_api_cost_usd',
            '>',
            100,
            AlertLevel.WARNING,
            "AI API cost high: ${value} (budget: ${threshold})"
        )

    def get_dashboard_data(self) -> Dict:
        """대시보드용 데이터"""

        return {
            'current_metrics': {
                name: values[-1] if values else None
                for name, values in self.metrics.items()
            },
            'recent_alerts': self.alert_history[-10:],
            'system_status': self._calculate_system_status()
        }

    def _calculate_system_status(self) -> str:
        """시스템 상태 계산"""

        recent_alerts = self.alert_history[-10:]

        critical_count = sum(1 for a in recent_alerts if a['level'] == AlertLevel.CRITICAL)
        error_count = sum(1 for a in recent_alerts if a['level'] == AlertLevel.ERROR)

        if critical_count > 0:
            return 'critical'
        elif error_count > 2:
            return 'degraded'
        else:
            return 'operational'
```

---

## 5. AI 모델 리스크 관리

### 5.1. AI 판단 검증 시스템

```python
from typing import Dict, List, Optional
import json

class AIDecisionValidator:
    """AI 의사결정 검증 시스템"""

    def __init__(self):
        self.validation_rules = []
        self.decision_history = []
        self.performance_metrics = {}

    def validate_ai_decision(self, decision: Dict, market_context: Dict) -> Dict:
        """AI 의사결정 검증"""

        validation_results = {
            'decision': decision,
            'checks': [],
            'approved': True,
            'confidence_adjustment': 0
        }

        # 1. 신뢰도 검증
        confidence_check = self._validate_confidence(decision)
        validation_results['checks'].append(confidence_check)

        # 2. 논리적 일관성 검증
        consistency_check = self._validate_consistency(decision, market_context)
        validation_results['checks'].append(consistency_check)

        # 3. 과거 성과 기반 검증
        performance_check = self._validate_based_on_performance(decision)
        validation_results['checks'].append(performance_check)

        # 4. 시장 상황 적합성 검증
        market_check = self._validate_market_conditions(decision, market_context)
        validation_results['checks'].append(market_check)

        # 5. 극단적 판단 검증
        extreme_check = self._check_extreme_decision(decision)
        validation_results['checks'].append(extreme_check)

        # 최종 승인 여부
        validation_results['approved'] = all(
            check['passed'] for check in validation_results['checks']
        )

        # 신뢰도 조정
        failed_checks = sum(1 for check in validation_results['checks'] if not check['passed'])
        validation_results['confidence_adjustment'] = -0.1 * failed_checks

        # 기록 저장
        self.decision_history.append({
            'timestamp': datetime.utcnow(),
            'decision': decision,
            'validation': validation_results
        })

        return validation_results

    def _validate_confidence(self, decision: Dict) -> Dict:
        """신뢰도 검증"""

        confidence = decision.get('confidence', 0)

        if confidence < 0.5:
            return {
                'check': 'confidence',
                'passed': False,
                'reason': f'Confidence too low: {confidence}'
            }

        if confidence > 0.95:
            # 과도한 확신도 의심스러움
            return {
                'check': 'confidence',
                'passed': False,
                'reason': f'Suspiciously high confidence: {confidence}'
            }

        return {'check': 'confidence', 'passed': True}

    def _validate_consistency(self, decision: Dict, market_context: Dict) -> Dict:
        """논리적 일관성 검증"""

        # 예: 가격 상승 중인데 SELL 신호
        if market_context.get('trend') == 'up' and decision.get('action') == 'SELL':
            if 'contrarian' not in decision.get('reasoning', '').lower():
                return {
                    'check': 'consistency',
                    'passed': False,
                    'reason': 'Inconsistent with market trend without contrarian reasoning'
                }

        return {'check': 'consistency', 'passed': True}

    def _validate_based_on_performance(self, decision: Dict) -> Dict:
        """과거 성과 기반 검증"""

        # 유사한 과거 결정 찾기
        similar_decisions = self._find_similar_decisions(decision)

        if len(similar_decisions) >= 5:
            success_rate = sum(1 for d in similar_decisions if d.get('outcome') == 'success') / len(similar_decisions)

            if success_rate < 0.3:
                return {
                    'check': 'performance',
                    'passed': False,
                    'reason': f'Poor historical performance: {success_rate:.1%} success rate'
                }

        return {'check': 'performance', 'passed': True}

    def _check_extreme_decision(self, decision: Dict) -> Dict:
        """극단적 판단 검증"""

        # 포지션 크기가 너무 큰 경우
        if decision.get('position_size_pct', 0) > 0.1:  # 10% 초과
            return {
                'check': 'extreme',
                'passed': False,
                'reason': 'Position size too large'
            }

        # Stop-loss가 너무 넓은 경우
        if decision.get('stop_loss_pct', 0) > 0.15:  # 15% 초과
            return {
                'check': 'extreme',
                'passed': False,
                'reason': 'Stop-loss too wide'
            }

        return {'check': 'extreme', 'passed': True}

    def track_decision_outcome(self, decision_id: str, outcome: Dict):
        """의사결정 결과 추적"""

        # 결정 찾기
        for decision in self.decision_history:
            if decision.get('id') == decision_id:
                decision['outcome'] = outcome
                break

        # 성과 메트릭 업데이트
        self._update_performance_metrics()

    def _update_performance_metrics(self):
        """성과 메트릭 업데이트"""

        completed_decisions = [d for d in self.decision_history if 'outcome' in d]

        if completed_decisions:
            self.performance_metrics = {
                'total_decisions': len(completed_decisions),
                'success_rate': sum(1 for d in completed_decisions if d['outcome'].get('success')) / len(completed_decisions),
                'avg_return': np.mean([d['outcome'].get('return_pct', 0) for d in completed_decisions]),
                'validation_effectiveness': self._calculate_validation_effectiveness()
            }

    def _calculate_validation_effectiveness(self) -> float:
        """검증 시스템 효과성 계산"""

        # 검증 통과한 결정 중 성공률 vs 전체 성공률 비교
        validated_decisions = [
            d for d in self.decision_history
            if d.get('validation', {}).get('approved') and 'outcome' in d
        ]

        if not validated_decisions:
            return 0

        validated_success_rate = sum(
            1 for d in validated_decisions if d['outcome'].get('success')
        ) / len(validated_decisions)

        return validated_success_rate
```

### 5.2. 모델 드리프트 감지

```python
import numpy as np
from scipy import stats
from typing import Dict, List

class ModelDriftDetector:
    """AI 모델 성능 저하 감지"""

    def __init__(self, baseline_window: int = 100, detection_threshold: float = 0.05):
        self.baseline_window = baseline_window
        self.detection_threshold = detection_threshold
        self.predictions = []
        self.actuals = []
        self.drift_alerts = []

    def add_prediction(self, prediction: Dict, actual: Dict):
        """예측과 실제 결과 추가"""

        self.predictions.append(prediction)
        self.actuals.append(actual)

        # 윈도우 크기 제한
        if len(self.predictions) > self.baseline_window * 2:
            self.predictions = self.predictions[-self.baseline_window * 2:]
            self.actuals = self.actuals[-self.baseline_window * 2:]

    def detect_drift(self) -> Dict:
        """드리프트 감지"""

        if len(self.predictions) < self.baseline_window * 1.5:
            return {'drift_detected': False, 'reason': 'Insufficient data'}

        # 기준 기간과 최근 기간 분리
        baseline_preds = self.predictions[:self.baseline_window]
        recent_preds = self.predictions[-self.baseline_window:]

        baseline_actuals = self.actuals[:self.baseline_window]
        recent_actuals = self.actuals[-self.baseline_window:]

        # 1. 정확도 드리프트
        accuracy_drift = self._detect_accuracy_drift(
            baseline_preds, baseline_actuals,
            recent_preds, recent_actuals
        )

        # 2. 분포 드리프트 (KS Test)
        distribution_drift = self._detect_distribution_drift(
            baseline_preds, recent_preds
        )

        # 3. 예측 신뢰도 드리프트
        confidence_drift = self._detect_confidence_drift(
            baseline_preds, recent_preds
        )

        drift_detected = (
            accuracy_drift['detected'] or
            distribution_drift['detected'] or
            confidence_drift['detected']
        )

        if drift_detected:
            alert = {
                'timestamp': datetime.utcnow(),
                'accuracy_drift': accuracy_drift,
                'distribution_drift': distribution_drift,
                'confidence_drift': confidence_drift
            }
            self.drift_alerts.append(alert)

        return {
            'drift_detected': drift_detected,
            'accuracy_drift': accuracy_drift,
            'distribution_drift': distribution_drift,
            'confidence_drift': confidence_drift,
            'recommendation': self._get_drift_recommendation(drift_detected)
        }

    def _detect_accuracy_drift(self, baseline_preds, baseline_actuals,
                              recent_preds, recent_actuals) -> Dict:
        """정확도 드리프트 감지"""

        # 정확도 계산
        baseline_accuracy = self._calculate_accuracy(baseline_preds, baseline_actuals)
        recent_accuracy = self._calculate_accuracy(recent_preds, recent_actuals)

        accuracy_change = abs(recent_accuracy - baseline_accuracy)

        return {
            'detected': accuracy_change > self.detection_threshold,
            'baseline_accuracy': baseline_accuracy,
            'recent_accuracy': recent_accuracy,
            'change': accuracy_change
        }

    def _detect_distribution_drift(self, baseline_preds, recent_preds) -> Dict:
        """분포 드리프트 감지 (Kolmogorov-Smirnov Test)"""

        baseline_values = [p.get('confidence', 0) for p in baseline_preds]
        recent_values = [p.get('confidence', 0) for p in recent_preds]

        ks_statistic, p_value = stats.ks_2samp(baseline_values, recent_values)

        return {
            'detected': p_value < self.detection_threshold,
            'ks_statistic': ks_statistic,
            'p_value': p_value
        }

    def _detect_confidence_drift(self, baseline_preds, recent_preds) -> Dict:
        """신뢰도 드리프트 감지"""

        baseline_conf = np.mean([p.get('confidence', 0) for p in baseline_preds])
        recent_conf = np.mean([p.get('confidence', 0) for p in recent_preds])

        conf_change = abs(recent_conf - baseline_conf)

        return {
            'detected': conf_change > 0.1,  # 10% 이상 변화
            'baseline_confidence': baseline_conf,
            'recent_confidence': recent_conf,
            'change': conf_change
        }

    def _calculate_accuracy(self, predictions, actuals) -> float:
        """정확도 계산"""

        correct = sum(
            1 for pred, actual in zip(predictions, actuals)
            if pred.get('action') == actual.get('best_action')
        )

        return correct / len(predictions) if predictions else 0

    def _get_drift_recommendation(self, drift_detected: bool) -> str:
        """드리프트 대응 권고사항"""

        if not drift_detected:
            return "No action needed"

        recommendations = [
            "1. Review recent model predictions for anomalies",
            "2. Check if market regime has changed",
            "3. Consider retraining the model with recent data",
            "4. Temporarily reduce position sizes",
            "5. Increase monitoring frequency"
        ]

        return "\n".join(recommendations)
```

---

## 6. 운영 리스크 관리

### 6.1. 운영 절차 및 체크리스트

```python
class OperationalRiskManager:
    """운영 리스크 관리"""

    def __init__(self):
        self.daily_checklist = []
        self.incident_log = []
        self.maintenance_schedule = []

    def create_daily_checklist(self) -> List[Dict]:
        """일일 운영 체크리스트"""

        return [
            {
                'task': 'Check system health',
                'priority': 'high',
                'responsible': 'operator',
                'checks': [
                    'API connections active',
                    'Database accessible',
                    'Sufficient disk space (>20%)',
                    'Memory usage normal (<80%)'
                ]
            },
            {
                'task': 'Review overnight trades',
                'priority': 'high',
                'responsible': 'trader',
                'checks': [
                    'All trades logged correctly',
                    'No unexpected losses',
                    'Stop-losses triggered appropriately'
                ]
            },
            {
                'task': 'Check AI performance',
                'priority': 'medium',
                'responsible': 'ml_engineer',
                'checks': [
                    'Model predictions within normal range',
                    'API costs within budget',
                    'No drift detected'
                ]
            },
            {
                'task': 'Verify risk limits',
                'priority': 'high',
                'responsible': 'risk_manager',
                'checks': [
                    'Position sizes within limits',
                    'Daily loss within threshold',
                    'Correlation risk acceptable'
                ]
            },
            {
                'task': 'Backup verification',
                'priority': 'medium',
                'responsible': 'operator',
                'checks': [
                    'Database backup completed',
                    'Configuration backup current',
                    'Recovery test passed'
                ]
            }
        ]

    def log_incident(self, incident: Dict):
        """인시던트 로깅"""

        incident_record = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow(),
            'severity': incident.get('severity', 'medium'),
            'category': incident.get('category'),
            'description': incident.get('description'),
            'impact': incident.get('impact'),
            'resolution': incident.get('resolution'),
            'root_cause': incident.get('root_cause'),
            'prevention': incident.get('prevention')
        }

        self.incident_log.append(incident_record)

        # 심각도가 높은 경우 즉시 알림
        if incident_record['severity'] == 'critical':
            self._send_critical_incident_alert(incident_record)

    def schedule_maintenance(self, maintenance: Dict):
        """정기 점검 일정"""

        self.maintenance_schedule.append({
            'scheduled_date': maintenance['date'],
            'duration_hours': maintenance['duration'],
            'type': maintenance['type'],  # 'full', 'partial'
            'components': maintenance['components'],
            'notification_sent': False
        })

    def pre_maintenance_checklist(self) -> List[str]:
        """점검 전 체크리스트"""

        return [
            "1. 24시간 전 사용자 공지",
            "2. 모든 포지션 청산 또는 헤지",
            "3. 데이터베이스 전체 백업",
            "4. 설정 파일 백업",
            "5. Rollback 계획 준비",
            "6. 비상 연락망 확인",
            "7. 테스트 환경 준비",
            "8. 점검 중 모니터링 시스템 설정"
        ]

    def generate_incident_report(self, start_date: datetime, end_date: datetime) -> Dict:
        """인시던트 리포트 생성"""

        period_incidents = [
            inc for inc in self.incident_log
            if start_date <= inc['timestamp'] <= end_date
        ]

        if not period_incidents:
            return {'period': f"{start_date} to {end_date}", 'incidents': 0}

        # 통계 계산
        by_severity = {}
        by_category = {}
        total_downtime = 0

        for incident in period_incidents:
            # 심각도별
            severity = incident['severity']
            by_severity[severity] = by_severity.get(severity, 0) + 1

            # 카테고리별
            category = incident['category']
            by_category[category] = by_category.get(category, 0) + 1

        return {
            'period': f"{start_date.date()} to {end_date.date()}",
            'total_incidents': len(period_incidents),
            'by_severity': by_severity,
            'by_category': by_category,
            'mtbf': self._calculate_mtbf(period_incidents),  # Mean Time Between Failures
            'mttr': self._calculate_mttr(period_incidents),  # Mean Time To Recovery
            'top_root_causes': self._get_top_root_causes(period_incidents)
        }

    def _calculate_mtbf(self, incidents: List[Dict]) -> float:
        """평균 장애 간격 계산"""

        if len(incidents) < 2:
            return 0

        incidents_sorted = sorted(incidents, key=lambda x: x['timestamp'])
        intervals = []

        for i in range(1, len(incidents_sorted)):
            interval = (incidents_sorted[i]['timestamp'] - incidents_sorted[i-1]['timestamp']).total_seconds() / 3600
            intervals.append(interval)

        return np.mean(intervals) if intervals else 0
```

---

## 7. 규제 및 컴플라이언스

### 7.1. 규제 준수 모니터링

```python
class ComplianceManager:
    """규제 준수 관리"""

    def __init__(self):
        self.regulations = {}
        self.compliance_checks = []
        self.audit_trail = []

    def setup_regulations(self, jurisdiction: str):
        """관할권별 규제 설정"""

        if jurisdiction == 'korea':
            self.regulations = {
                'kyc_required': True,
                'max_leverage': 1.0,  # 레버리지 금지
                'tax_rate': 0.22,  # 22% (지방세 포함)
                'reporting_threshold': 50000000,  # 5천만원
                'travel_rule': True  # 트래블룰 적용
            }
        elif jurisdiction == 'usa':
            self.regulations = {
                'pattern_day_trader_rule': True,
                'wash_sale_rule': True,
                'min_account_balance': 25000,  # PDT rule
                'tax_rate': 0.15,  # Long-term capital gains
                'form_8949_required': True
            }

    def check_compliance(self, trade: Dict) -> Dict:
        """거래 컴플라이언스 체크"""

        violations = []

        # KYC 체크
        if self.regulations.get('kyc_required'):
            if not trade.get('kyc_verified'):
                violations.append('KYC not verified')

        # 레버리지 체크
        if trade.get('leverage', 1) > self.regulations.get('max_leverage', 1):
            violations.append(f"Leverage {trade['leverage']} exceeds limit")

        # Pattern Day Trader Rule (미국)
        if self.regulations.get('pattern_day_trader_rule'):
            if self._is_pattern_day_trader() and self._get_account_balance() < 25000:
                violations.append('PDT rule violation')

        return {
            'compliant': len(violations) == 0,
            'violations': violations
        }

    def calculate_tax_liability(self, trades: List[Dict]) -> Dict:
        """세금 계산"""

        total_gains = 0
        total_losses = 0

        for trade in trades:
            if trade['pnl'] > 0:
                total_gains += trade['pnl']
            else:
                total_losses += abs(trade['pnl'])

        net_gains = total_gains - total_losses

        # 세금 계산
        tax_rate = self.regulations.get('tax_rate', 0)
        tax_liability = max(0, net_gains * tax_rate)

        return {
            'total_gains': total_gains,
            'total_losses': total_losses,
            'net_gains': net_gains,
            'tax_rate': tax_rate,
            'tax_liability': tax_liability,
            'required_forms': self._get_required_tax_forms()
        }

    def create_audit_trail(self, action: Dict):
        """감사 추적 생성"""

        audit_entry = {
            'timestamp': datetime.utcnow(),
            'action': action['type'],
            'user': action.get('user', 'system'),
            'details': action['details'],
            'ip_address': action.get('ip_address'),
            'result': action.get('result'),
            'hash': self._calculate_hash(action)
        }

        self.audit_trail.append(audit_entry)

        # 변조 방지를 위한 체인 해시
        if len(self.audit_trail) > 1:
            prev_hash = self.audit_trail[-2]['hash']
            audit_entry['prev_hash'] = prev_hash
            audit_entry['hash'] = self._calculate_hash(audit_entry)

    def generate_compliance_report(self) -> Dict:
        """컴플라이언스 리포트 생성"""

        return {
            'generated_at': datetime.utcnow(),
            'jurisdiction': self.regulations,
            'total_trades': len(self.audit_trail),
            'compliance_rate': self._calculate_compliance_rate(),
            'violations': self._get_recent_violations(),
            'tax_summary': self.calculate_tax_liability(self._get_trades_from_audit()),
            'audit_integrity': self._verify_audit_trail_integrity()
        }

    def _calculate_hash(self, data: Dict) -> str:
        """해시 계산"""
        import hashlib
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _verify_audit_trail_integrity(self) -> bool:
        """감사 추적 무결성 검증"""

        for i in range(1, len(self.audit_trail)):
            entry = self.audit_trail[i]

            # 이전 해시 검증
            if 'prev_hash' in entry:
                if entry['prev_hash'] != self.audit_trail[i-1]['hash']:
                    return False

            # 현재 해시 검증
            recalculated_hash = self._calculate_hash({
                k: v for k, v in entry.items() if k != 'hash'
            })

            if recalculated_hash != entry['hash']:
                return False

        return True
```

---

## 문서 요약

이 리스크 관리 전략 문서는 AI 가상화폐 자동매매 시스템의 모든 리스크 측면을 다룹니다:

### 주요 리스크 관리 영역

1. **자금 관리**
   - Kelly Criterion 기반 포지션 크기 결정
   - Core-Satellite 및 리스크 패리티 전략
   - 동적 자금 배분

2. **포지션 리스크**
   - 다양한 Stop-Loss 전략 (고정, Trailing, ATR, 시간 기반)
   - 포지션 상관관계 관리
   - 포트폴리오 VaR 계산

3. **시스템 리스크**
   - 재해 복구 시스템 (Failover/Failback)
   - 실시간 모니터링 및 알림
   - 데이터 동기화 및 백업

4. **AI 모델 리스크**
   - 의사결정 검증 시스템
   - 모델 드리프트 감지
   - 성과 기반 검증

5. **운영 리스크**
   - 일일 체크리스트
   - 인시던트 관리
   - 정기 점검 절차

6. **규제 컴플라이언스**
   - 관할권별 규제 준수
   - 세금 계산 및 보고
   - 감사 추적 관리

### 리스크 한도 요약

- **최대 포지션 크기**: 총 자산의 5%
- **최대 총 노출**: 총 자산의 30%
- **일일 손실 한도**: 5%
- **최대 드로다운**: 15%
- **최소 AI 신뢰도**: 70%

이러한 종합적인 리스크 관리 체계를 통해 안정적이고 지속 가능한 자동매매 시스템을 운영할 수 있습니다.