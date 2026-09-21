import contextlib
import datetime
import io
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import config
from database import db_manager as db
from risk.risk_manager import RiskManager
from scanner import scoring_engine as scoring
from scanner import market_analyzer as analyzer
from scanner import news_filter as news

class TradingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dbpatch = patch.object(config,'DB_PATH',str(Path(self.temp.name)/'scanner.db'))
        self.dbpatch.start()
        self.addCleanup(self.dbpatch.stop)
        db.init_db()
        self.cache = dict(news._cache)
        self.addCleanup(lambda:news._cache.update(self.cache))

    def bars(self, n=60):
        close = np.linspace(100,120,n)
        return pd.DataFrame({'Open':close-.2,'High':close+.5,'Low':close-.5,'Close':close})


    def test_invalid_account_snapshot_is_blocked(self):
        risk = RiskManager()
        for balance,equity in ((0,100),(float('nan'),100),(100,float('inf')),(None,100)):
            self.assertFalse(risk.check_risk(balance,equity)['allowed'])
        self.assertIsNone(risk._daily_start)
        self.assertTrue(risk.check_risk(10000,10000)['allowed'])

    def test_weekly_baseline_resets_on_new_week(self):
        with patch('risk.risk_manager.date') as clock:
            clock.today.return_value = datetime.date(2026,7,3)
            risk = RiskManager()
            risk.check_risk(10000,10000)
            clock.today.return_value = datetime.date(2026,7,6)
            status = risk.check_risk(12000,12000)
        self.assertEqual(status['weekly_start'],12000)
        self.assertEqual(status['weekly_drawdown'],0)

    def test_invalid_lot_inputs_do_not_allocate_risk(self):
        risk = RiskManager()
        for balance,entry,stop,pip in ((10000,1.1,1.1,10),(float('nan'),1.1,1.0,10),(10000,1.1,1.0,0)):
            self.assertEqual(risk.calculate_position_size(balance,entry,stop,pip),0)

    def test_four_and_five_digit_sizing_agree(self):
        risk = RiskManager()
        self.assertEqual(risk.calculate_position_size(10000,1.1,1.09,digits=4),
                         risk.calculate_position_size(10000,1.1,1.09,digits=5))
        self.assertEqual(risk.calculate_position_size(10000,150,149,digits=2),
                         risk.calculate_position_size(10000,150,149,digits=3))

    def test_lot_floor_never_exceeds_configured_risk(self):
        risk = RiskManager()
        lots = risk.calculate_position_size(1000,1.1,1.087,digits=5)
        self.assertLessEqual(lots*130*10,1000*config.MAX_RISK_PER_TRADE/100)
        self.assertEqual(risk.calculate_position_size(1,1.1,1.0),0)

    def test_invalid_risk_configuration_fails_startup(self):
        for value in (0,-1,float('nan'),101):
            with patch.object(config,'MAX_RISK_PER_TRADE',value), self.assertRaises(ValueError):
                RiskManager()
        with patch.object(config,'MAX_CONSECUTIVE_LOSSES',0), self.assertRaises(ValueError):
            RiskManager()

    def test_elite_setup_requires_boolean_approval(self):
        risk = RiskManager()
        for approval in ('false','true',1,None):
            self.assertFalse(risk.is_elite_trade_allowed({'grade':'ELITE'}, {'allowed':approval,'daily_drawdown':0}))
        self.assertTrue(risk.is_elite_trade_allowed({'grade':'ELITE'}, {'allowed':True,'daily_drawdown':0}))

    def test_latest_win_breaks_same_timestamp_loss_streak(self):
        loss = db.log_trade('EURUSD','buy',1.1,1.0,1.2)
        win = db.log_trade('EURUSD','buy',1.1,1.0,1.2)
        db.close_trade(loss,1.0,'loss')
        db.close_trade(win,1.2,'win')
        with db.get_connection() as conn:
            conn.execute("UPDATE trades SET closed_at='2026-07-01 12:00:00'")
        self.assertEqual(db.get_recent_losses(),[])

    def test_empty_daily_stats_are_numeric(self):
        self.assertEqual(db.get_daily_stats(),{'total':0,'wins':0,'losses':0,'avg_rr':0})

    def test_weekly_stats_date_window(self):
        with db.get_connection() as conn:
            for opened in ('2026-07-01','2026-07-02','2026-07-08','2026-07-09'):
                conn.execute("INSERT INTO trades(symbol,opened_at,outcome) VALUES('EURUSD',?,'win')",(opened,))
        with patch.object(db,'date') as clock:
            clock.today.return_value = datetime.date(2026,7,8)
            self.assertEqual(db.get_weekly_stats()['total'],2)

    def test_repeated_trade_close_preserves_first_outcome(self):
        tid = db.log_trade('EURUSD','buy',1.1,1.0,1.2)
        self.assertTrue(db.close_trade(tid,1.2,'win'))
        self.assertFalse(db.close_trade(tid,1.0,'loss'))
        self.assertEqual(db.get_all_trades()[0]['outcome'],'win')
        self.assertFalse(db.close_trade(999,1.0,'loss'))

    def test_failed_setup_write_closes_database_connection(self):
        connection = db.get_connection()
        wrapped = MagicMock(wraps=connection)
        with patch.object(db,'get_connection',return_value=wrapped):
            with self.assertRaises(TypeError):
                db.save_setup('EURUSD','M15',{'conditions':[object()]},{'score':4})
        wrapped.close.assert_called_once()
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute('SELECT 1')

    def test_orphan_trade_reference_is_rejected(self):
        with self.assertRaises(sqlite3.IntegrityError):
            db.log_trade('EURUSD','buy',1.1,1.0,1.2,setup_id=999)
        self.assertEqual(db.get_all_trades(),[])

    def test_nested_database_path_initializes(self):
        path = Path(self.temp.name)/'nested'/'scanner.db'
        with patch.object(config,'DB_PATH',str(path)):
            db.init_db()
            self.assertEqual(db.get_all_trades(),[])
        self.assertTrue(path.exists())

    def test_unknown_bias_gets_no_directional_points(self):
        self.assertEqual(scoring.score_setup({'bias':None})['score'],0)
        self.assertEqual(scoring.score_setup({'bias':'bullish'})['score'],2)

    def test_unknown_fvg_fill_state_does_not_add_points(self):
        self.assertEqual(scoring.score_setup({'fvgs':[{},None,{'filled':True}]})['score'],0)
        self.assertEqual(scoring.score_setup({'fvgs':[{'filled':False}]})['score'],2)

    def test_invalid_reward_ratio_cannot_improve_grade(self):
        for ratio in (None,'3',float('inf'),float('nan')):
            self.assertEqual(scoring.score_setup({'levels':{'rr':ratio}})['score'],0)
        self.assertEqual(scoring.score_setup({'levels':{'rr':2}})['score'],1)

    def test_empty_and_nonfinite_feeds_are_neutral(self):
        valid = self.bars()
        for invalid in (pd.DataFrame(),valid.assign(Close=float('nan'))):
            result = analyzer.analyze_symbol(valid,valid,invalid,'EURUSD')
            self.assertEqual(result['bias'],'neutral')
            self.assertIsNone(result['levels'])
        self.assertEqual(analyzer.detect_fvg(pd.DataFrame()),[])

    def test_gap_remains_filled_after_recovery(self):
        bars = pd.DataFrame({'Open':[10,11,14,10,14],'High':[11,12,15,12,15],
                             'Low':[9,10,13,10,13],'Close':[10,11,14,11,14]})
        gaps = analyzer.detect_fvg(bars)
        initial = next(g for g in gaps if g['index']==2)
        self.assertTrue(initial['filled'])

    def test_stale_swing_levels_cannot_invert_stop_direction(self):
        bars = self.bars()
        for bias, highs, lows in (('bullish',[],[(1,150)]),('bearish',[(1,90)],[])):
            levels = analyzer.calculate_entry_levels(bars,bias,highs,lows)
            if bias=='bullish':
                self.assertLess(levels['sl'],levels['entry'])
            else:
                self.assertGreater(levels['sl'],levels['entry'])
            self.assertAlmostEqual(abs(levels['entry']-levels['sl']),levels['risk'],places=4)

    def test_bad_calendar_payload_preserves_valid_cache(self):
        cached = [{'title':'CPI','date':'2026-07-01T12:00:00Z'}]
        news._cache.update(events=cached,fetched_at=None)
        response = MagicMock()
        response.json.return_value = {'error':'upstream failed'}
        with patch.object(news.requests,'get',return_value=response), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(news.fetch_calendar(force=True),cached)
        self.assertIsNone(news._cache['fetched_at'])

    def test_empty_calendar_does_not_refetch_each_scan(self):
        news._cache.update(events=[],fetched_at=datetime.datetime.now(datetime.timezone.utc))
        with patch.object(news.requests,'get') as fetch:
            self.assertEqual(news.fetch_calendar(),[])
        fetch.assert_not_called()

    def test_unavailable_calendar_blocks_news_sensitive_symbol(self):
        news._cache.update(events=[],fetched_at=None)
        with patch.object(news.requests,'get',side_effect=OSError('offline')), contextlib.redirect_stdout(io.StringIO()):
            blocked, reason = news.is_news_blackout('EURUSD')
        self.assertTrue(blocked)
        self.assertIn('unavailable',reason)
