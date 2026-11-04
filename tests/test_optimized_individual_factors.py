"""Individual signal contracts; no account or GA execution."""
import unittest
from types import SimpleNamespace
import numpy as np
from factor.optimized import OPTIMIZED_FACTORS, get_optimized_factor_definition
from factor.compute import _factor_panel
from factor.library.optimized_eight import (
    TrueMarketCap, VolumeCV, AmountBasedSmallCap5, PB3ROE5Signal,
    ProfitGrowthOriginalAvailability, BiliLowReturnAttention10, LowResidualVol20,
)


class OptimizedFactorTests(unittest.TestCase):
    def test_catalog_lag_and_defaults(self):
        from factor.registry import PRODUCTION_FACTOR_NAMES
        self.assertNotIn("PB3ROE5Signal", PRODUCTION_FACTOR_NAMES)
        self.assertEqual(len(OPTIMIZED_FACTORS), 8)
        for name in ("TrueMarketCap", "PB3ROE5Signal"):
            d=get_optimized_factor_definition(name)
            runtime=SimpleNamespace(data={k:np.ones((4,2)) for k in d.metadata.required_fields}, trade_dates=np.arange(4))
            runtime.data["total_share"]=np.arange(8.).reshape(4,2)
            panel=_factor_panel(runtime,d)
            self.assertTrue(np.isnan(panel["total_share"][0]).all())
            np.testing.assert_array_equal(panel["total_share"][1:],runtime.data["total_share"][:-1])

    def test_pb_wan_units_and_strict_thresholds(self):
        panel={"open":np.array([[15.,14.,10.,10.,10.]]),"total_share":np.array([[10000.]*5]),
               "financial_equity":np.array([[5e8]*4+[np.nan]]),"financial_profit_ttm":np.array([[5e7,5e7,2.5e7,2.6e7,5e7]])}
        np.testing.assert_equal(PB3ROE5Signal().calc_batch(panel),[[0.,1.,0.,1.,np.nan]])
        out=ProfitGrowthOriginalAvailability().calc_batch({"financial_profit_yoy":np.array([[0.,2.,3.]]),"financial_cash_outflow_yoy":np.array([[1.,np.nan,np.inf]])})
        np.testing.assert_equal(out,[[0.,np.nan,np.nan]])

    def test_live_gate_and_completed_windows(self):
        panel={"open":np.full((25,3),10.),"st_mask":np.zeros((25,3)),"total_share":np.full((25,3),100.),"volume":np.tile(np.arange(1.,26.)[:,None],(1,3)),"amount":np.full((25,3),1e8)}
        panel["open"][24,0]=1.;panel["st_mask"][24,1]=1.
        for cls in (TrueMarketCap,VolumeCV,AmountBasedSmallCap5):
            out=cls().calc_batch(panel)
            self.assertTrue(np.isnan(out[24,:2]).all());self.assertTrue(np.isfinite(out[24,2]))
        self.assertAlmostEqual(AmountBasedSmallCap5().calc_batch(panel)[20,2],100*np.exp(-.2))
        self.assertAlmostEqual(VolumeCV().calc_batch(panel)[20,2],-np.std(np.arange(1.,21.))/10.5)
        original=AmountBasedSmallCap5().calc_batch(panel)[20].copy()
        panel["amount"][20:]=1e12
        np.testing.assert_equal(AmountBasedSmallCap5().calc_batch(panel)[20],original)

    def test_attention_and_residual_against_direct_windows(self):
        rng=np.random.default_rng(31);returns=rng.normal(0,.02,(35,4))
        panel={"close":100*(1+returns),"preClose":np.full((35,4),100.),"listing_age":np.ones((35,4)),"delisted_mask":np.zeros((35,4))}
        panel["listing_age"][:15,3]=-1
        market=np.array([returns[t,:3].mean() if t<15 else returns[t].mean() for t in range(35)])
        residual=returns-market[:,None];residual[:15,3]=np.nan
        for cls,w,stat in ((BiliLowReturnAttention10,10,lambda a:-np.max(np.abs(a),axis=0)),(LowResidualVol20,20,lambda a:-np.std(a,axis=0))):
            actual=cls().calc_batch(panel)
            self.assertTrue(np.isnan(actual[:w]).all())
            for t in range(w,35):np.testing.assert_allclose(actual[t],stat(residual[t-w:t]),rtol=2e-6,atol=1e-8,equal_nan=True)
            changed={k:v.copy() for k,v in panel.items()};changed["close"][25:]*=3
            np.testing.assert_equal(cls().calc_batch(changed)[:26],actual[:26])


if __name__=="__main__":unittest.main()
