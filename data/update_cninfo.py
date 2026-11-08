"""CNINFO HTTPS with TTL-bound public DNS and original-host TLS verification."""
import ipaddress
import logging
import time
from urllib.parse import urlsplit, urlunsplit

import requests
from requests.adapters import HTTPAdapter
import dns.resolver

HOST = 'webapi.cninfo.com.cn'


class HostTLSAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **kwargs):
        super().init_poolmanager(connections, maxsize, block=block,
                                 assert_hostname=HOST, server_hostname=HOST, **kwargs)


class CninfoSession:
    def __init__(self):
        self.session = requests.Session()
        self.session.trust_env = False
        self.address = None
        self.expires = 0
        self.resolved_at = 0
        self.verified_at = 0
        self.resolver = dns.resolver.Resolver(configure=False)
        self.resolver.nameservers = ['119.29.29.29', '223.5.5.5']

    def post(self, url, **kwargs):
        parts = urlsplit(url)
        if parts.scheme != 'https' or parts.hostname != HOST:
            raise ValueError('CNINFO transport only accepts its original HTTPS origin')
        if kwargs.get('verify', True) is False:
            raise ValueError('CNINFO TLS verification cannot be disabled')
        if time.monotonic() >= self.expires:
            try:
                answer = self.resolver.resolve(HOST, 'A', lifetime=5, search=False)
            except dns.exception.DNSException:
                # CNINFO intermittently returns NODATA. Keep a recently resolved,
                # already HTTPS-verified endpoint briefly; TLS is checked again.
                now = time.monotonic()
                if self.address is None or not self.verified_at or now - self.resolved_at > 3600:
                    raise
                self.expires = min(now + 30, self.resolved_at + 3600)
                logging.getLogger(__name__).warning('CNINFO DNS unavailable; reusing verified HTTPS endpoint for 30 seconds')
            else:
                address = ipaddress.IPv4Address(answer[0].address)
                ttl = answer.rrset.ttl
                if not address.is_global or ttl <= 0:
                    raise ValueError('CNINFO fallback DNS address/TTL is invalid')
                if str(address) != self.address:
                    self.session.mount('https://' + str(address) + '/', HostTLSAdapter())
                    self.verified_at = 0
                self.address = str(address)
                self.resolved_at = time.monotonic()
                self.expires = self.resolved_at + min(ttl, 60)
        headers = dict(kwargs.pop('headers', {}), Host=HOST)
        resolved = urlunsplit((parts.scheme, self.address, parts.path, parts.query, parts.fragment))
        response = self.session.post(resolved, headers=headers, **kwargs)
        if response.status_code == 200:
            self.verified_at = time.monotonic()
        return response
