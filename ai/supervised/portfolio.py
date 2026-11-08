"""Frozen predictions -> Qlib allocations -> the single WBR account engine."""
from __future__ import annotations
import numpy as np

from factor import FactorDefinition,FactorMetadata,precompute_factors
from env.action_schema import ActionSchema
from env.backtest import PreparedEpisode,EpisodeSession,run_target_weight_episode
from env.contracts import PortfolioExecutionSpec
from env.fees import DEFAULT_FEE_SCHEDULE
from env.qlib_strategy import RankBufferConfig,RankBufferPortfolio


class PreparedModelPortfolio:
    @classmethod
    def from_scores(cls,runtime,prediction,*,initial_cash,source_hash,split):
        """Already verified frozen signals, through the same native evaluator."""
        if prediction.shape!=(runtime.n_dates,runtime.n_stocks):raise ValueError('score axes differ')
        self=cls.__new__(cls);self.runtime=runtime;self.split=split;self.initial_cash=initial_cash
        self._set_predictions(runtime,prediction,'MultiHorizonScore',source_hash)
        return self

    def _set_predictions(self,rt,prediction,name,source_hash):
        self.prediction=prediction
        class PredictionFactor:
            def calc_batch(self,panel):
                if panel['open'] is not rt.field('open'):raise ValueError('prediction score used with different market snapshot')
                return prediction
        definition=FactorDefinition(FactorMetadata(name,'frozen-open-rank-prediction-v1',0,('open',),
            source_hash),PredictionFactor,raw_runtime_view=True)
        self.factors=precompute_factors(rt,definitions=(definition,))
        self.episode=PreparedEpisode.build(rt,self.factors,encode_observations=False)

    def policy(self, parameters):
        """One frozen allocation/config assembly, shared by every account day."""
        cfg=RankBufferConfig(**parameters)
        strategy=RankBufferPortfolio(self.runtime.stock_codes,cfg)
        schema=ActionSchema(factor_names=self.factors.factor_names,filter_names=self.factors.filter_names,
            fixed_buy_n=cfg.buy_n,fixed_sell_m=cfg.sell_m,
            fixed_turnover_rate=cfg.max_replacements/cfg.buy_n,
            fixed_rebalance_band_pct=.01,schema_version='supervised-portfolio-day-v1')
        return strategy,schema,schema.decode(np.ones(schema.action_dim)),PortfolioExecutionSpec(cfg.sell_m,1.,allow_cash=True)

    def decision_at(self, account, index, strategy):
        """The single strategy decision used by live and historical accounts."""
        market=self.episode.market_at(index)
        weights=strategy.targets(market,account,index,self.prediction[index])
        return market,weights

    def backtest_day(self, account, index, holding_days, parameters):
        """Compute today's backtest plan; the caller chooses simulated or real fills."""
        from env.planner import DayPlanner
        strategy,schema,config,spec=self.policy(parameters)
        if len(account.positions)>strategy.config.sell_m:
            raise ValueError('Account exceeds strategy capacity')
        if set(holding_days)!=set(account.positions):
            raise ValueError('Every holding requires a verified trading-day age')
        strategy.born={code:index-int(age) for code,age in holding_days.items()}
        market,weights=self.decision_at(account,index,strategy)
        plan=DayPlanner(fees=DEFAULT_FEE_SCHEDULE).plan(market,account,config,
            target_weights=weights,portfolio_spec=spec)
        return plan,market,weights

    def evaluate(self, parameters, *, details=False, on_transition=None):
        if self.episode.transition_count<1:
            raise ValueError('Settled backtest requires a next-open row; use backtest_day for today')
        strategy,schema,config,spec=self.policy(parameters)
        session=EpisodeSession(self.episode,action_schema=schema,initial_cash=self.initial_cash,
                               fees=DEFAULT_FEE_SCHEDULE,portfolio_spec=spec)
        def targets(context):
            _,weights=self.decision_at(context.current_account,context.current_index,strategy)
            return weights
        trace=run_target_weight_episode(session,config,targets,record_details=details,on_transition=on_transition)
        self.final_account=session.current_account
        metrics=trace.metrics.as_dict()
        metrics.update(mean_exposure=float(np.mean(trace.exposure)),mean_cash=float(np.mean(trace.cash)),
                       executed_sells=int(trace.executed_sell_count),allocation_contract_satisfied=bool(trace.full_investment_contract_satisfied),
                       full_investment_required=False,solver_status_counts={},funding_floor_days=0)
        if not details:
            metrics.update(total_fees=float(trace.total_fees),mean_gross_turnover=float(trace.sum_gross_turnover_ratio/len(trace.portfolio_returns)),
                           mean_total_cost=float(trace.sum_total_cost_ratio/len(trace.portfolio_returns)))
        return trace,metrics,strategy.diagnostics
