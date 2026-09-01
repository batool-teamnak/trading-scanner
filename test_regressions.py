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
