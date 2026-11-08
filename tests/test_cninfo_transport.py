from types import SimpleNamespace as NS

import pytest

from data.update_cninfo import CninfoSession, HOST


@pytest.mark.parametrize('address,ttl,valid', [('120.48.196.100', 10, True),
                                            ('127.0.0.1', 10, False),
                                            ('120.48.196.100', 0, False)])
def test_dns_fallback_preserves_tls_hostname_and_rejects_bad_answers(address, ttl, valid):
    client = CninfoSession()
    calls, adapters = [], []
    response = NS(status_code=200)
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return response
    class Answer(list):
        rrset = NS(ttl=ttl)
    answer = Answer([NS(address=address)])
    client.resolver = NS(resolve=lambda *a, **k: answer)
    client.session = NS(post=post, mount=lambda url, adapter: adapters.append(adapter))
    url = 'https://' + HOST + '/api/stock/p_stock2109'
    if not valid:
        with pytest.raises(ValueError):
            client.post(url, headers={'Accept-Enckey': 'public signature'}, timeout=15)
        assert len(calls) == 0
        return
    assert client.post(url, headers={'Accept-Enckey': 'public signature'}, timeout=15) is response
    assert calls[0][0] == 'https://120.48.196.100/api/stock/p_stock2109'
    assert calls[0][1]['headers']['Host'] == HOST
    assert 'verify' not in calls[0][1]
    assert adapters[0].poolmanager.connection_pool_kw['assert_hostname'] == HOST
    assert adapters[0].poolmanager.connection_pool_kw['server_hostname'] == HOST
    client.post(url, timeout=15)
    assert len(adapters) == 1


@pytest.mark.parametrize('verified,resolved,allowed', [(True, 1, True), (True, -3499.5, True),
                                                     (False, 1, False), (True, -4000, False)])
def test_dns_outage_only_reuses_recently_resolved_verified_endpoint(monkeypatch, verified, resolved, allowed):
    import dns.resolver
    import data.update_cninfo as module
    client = CninfoSession()
    client.address, client.resolved_at = '120.48.196.100', resolved
    client.verified_at = 2 if verified else 0
    monkeypatch.setattr(module.time, 'monotonic', lambda: 100)
    def fail(*args, **kwargs):
        raise dns.resolver.NoAnswer()
    client.resolver = NS(resolve=fail)
    calls = []
    client.session = NS(post=lambda *a, **k: calls.append((a, k)) or NS(status_code=200))
    if allowed:
        client.post('https://' + HOST + '/api/stock/p_stock2109')
        assert len(calls) == 1 and client.expires == min(130, resolved + 3600)
    else:
        with pytest.raises(dns.resolver.NoAnswer):
            client.post('https://' + HOST + '/api/stock/p_stock2109')
        assert not calls
