import socket
import time
import unittest
from typing import ClassVar
from unittest.mock import patch

from llama_index.core import Document

from utils import helpers, llama_index


class FakeResponse:
    def __init__(self, status=200, headers=None, chunks=()):
        self.status = status
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self.chunks = tuple(chunks)
        self.closed = False
        self.released = False

    def stream(self, amt=65536, decode_content=True):
        yield from self.chunks

    def close(self):
        self.closed = True

    def release_conn(self):
        self.released = True


class FakePool:
    instances: ClassVar[list] = []
    responses: ClassVar[list] = []

    def __init__(self, host, port, **kwargs):
        self.host = host
        self.port = port
        self.kwargs = kwargs
        self.calls = []
        self.closed = False
        self.__class__.instances.append(self)

    def urlopen(self, method, target, **kwargs):
        self.calls.append((method, target, kwargs))
        if not self.__class__.responses:
            raise AssertionError("No fake response was configured")
        return self.__class__.responses.pop(0)

    def close(self):
        self.closed = True


def dns_result(address, port=443):
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    return [(family, socket.SOCK_STREAM, 6, "", (address, port))]


class WebsiteAddressValidationTests(unittest.TestCase):
    def test_rejects_non_global_ipv4_and_ipv6_destinations(self):
        addresses = (
            "127.0.0.1",
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.1.1",
            "0.0.0.0",
            "224.0.0.1",
            "240.0.0.1",
            "100.64.0.1",
            "::1",
            "fc00::1",
            "fd12:3456::1",
            "fe80::1",
            "ff02::1",
            "::",
            "2001:db8::1",
            "::ffff:127.0.0.1",
        )
        for address in addresses:
            with self.subTest(address=address):
                with (
                    patch.object(
                        helpers.socket,
                        "getaddrinfo",
                        return_value=dns_result(address),
                    ),
                    self.assertRaises(helpers.WebsiteIngestionError) as context,
                ):
                    helpers.validate_website_urls(["https://example.com"])
                self.assertEqual(context.exception.category, "blocked_destination")

    def test_rejects_known_metadata_ip_even_when_it_is_global(self):
        with (
            patch.object(
                helpers.socket,
                "getaddrinfo",
                return_value=dns_result("168.63.129.16"),
            ),
            self.assertRaises(helpers.WebsiteIngestionError) as context,
        ):
            helpers.validate_website_urls(["https://metadata.example"])
        self.assertEqual(context.exception.category, "blocked_destination")

    def test_rejects_known_metadata_hostname_before_dns(self):
        with (
            patch.object(helpers.socket, "getaddrinfo") as resolver,
            self.assertRaises(helpers.WebsiteIngestionError) as context,
        ):
            helpers.validate_website_urls(["https://metadata.google.internal"])
        resolver.assert_not_called()
        self.assertEqual(context.exception.category, "blocked_destination")

    def test_accepts_global_ipv4_and_ipv6_results(self):
        for address in ("93.184.216.34", "2001:4860:4860::8888"):
            with (
                self.subTest(address=address),
                patch.object(
                    helpers.socket,
                    "getaddrinfo",
                    return_value=dns_result(address),
                ),
            ):
                self.assertEqual(
                    helpers.validate_website_urls(["https://example.com"]),
                    ["https://example.com"],
                )

    def test_rejects_a_hostname_with_any_blocked_address(self):
        with (
            patch.object(
                helpers.socket,
                "getaddrinfo",
                return_value=dns_result("93.184.216.34") + dns_result("192.168.1.1"),
            ),
            self.assertRaises(helpers.WebsiteIngestionError) as context,
        ):
            helpers.validate_website_urls(["https://example.com"])
        self.assertEqual(context.exception.category, "blocked_destination")

    def test_url_limit_is_checked_before_dns(self):
        urls = [f"https://example-{index}.com" for index in range(7)]
        with (
            patch.object(helpers.socket, "getaddrinfo") as resolver,
            self.assertRaises(helpers.WebsiteIngestionError) as context,
        ):
            helpers.validate_website_urls(urls)
        resolver.assert_not_called()
        self.assertEqual(context.exception.category, "url_limit")
        self.assertEqual(context.exception.diagnostic()["category"], "url_limit")

    def test_invalid_url_and_dns_failure_are_distinct(self):
        with self.assertRaises(helpers.WebsiteIngestionError) as invalid:
            helpers.validate_website_urls(["not-a-url"])
        self.assertEqual(invalid.exception.category, "invalid_url")
        with (
            patch.object(
                helpers.socket,
                "getaddrinfo",
                side_effect=socket.gaierror("not found"),
            ),
            self.assertRaises(helpers.WebsiteIngestionError) as dns,
        ):
            helpers.validate_website_urls(["https://example.com"])
        self.assertEqual(dns.exception.category, "dns_failure")
        with (
            patch.object(
                helpers.socket,
                "getaddrinfo",
                side_effect=TimeoutError("resolver timed out"),
            ),
            self.assertRaises(helpers.WebsiteIngestionError) as timeout,
        ):
            helpers.validate_website_urls(["https://example.com"])
        self.assertEqual(timeout.exception.category, "timeout")


class PinnedTransportTests(unittest.TestCase):
    def setUp(self):
        FakePool.instances = []
        FakePool.responses = []

    def test_validated_address_is_used_for_the_tls_connection(self):
        FakePool.responses = [FakeResponse(chunks=(b"<p>Readable page content</p>",))]
        with (
            patch.object(
                helpers.socket,
                "getaddrinfo",
                return_value=dns_result("93.184.216.34"),
            ),
            patch.object(helpers.urllib3, "HTTPSConnectionPool", FakePool),
        ):
            documents = helpers.load_website_documents(
                ["https://example.com/page"],
                deadline=time.monotonic() + 30,
            )
        self.assertEqual(len(documents), 1)
        self.assertEqual(len(FakePool.instances), 1)
        pool = FakePool.instances[0]
        self.assertEqual(pool.host, "93.184.216.34")
        self.assertEqual(pool.kwargs["assert_hostname"], "example.com")
        self.assertEqual(pool.kwargs["server_hostname"], "example.com")
        self.assertLessEqual(pool.calls[0][2]["timeout"].total, 30)
        self.assertLessEqual(pool.calls[0][2]["timeout"].read_timeout, 20)
        self.assertEqual(pool.calls[0][1], "/page")
        self.assertTrue(FakePool.responses == [])

    def test_dns_rebinding_cannot_redirect_the_connection_to_private_address(self):
        with (
            patch.object(
                helpers.socket,
                "getaddrinfo",
                side_effect=[
                    dns_result("93.184.216.34"),
                    dns_result("192.168.1.10"),
                ],
            ),
            patch.object(helpers.urllib3, "HTTPSConnectionPool", FakePool),
            self.assertRaises(helpers.WebsiteIngestionError) as context,
        ):
            helpers.load_website_documents(
                ["https://example.com"],
                deadline=time.monotonic() + 30,
            )
        self.assertEqual(context.exception.category, "blocked_destination")
        self.assertEqual(FakePool.instances, [])

    def test_redirect_to_blocked_destination_is_rejected_and_response_is_closed(self):
        redirect = FakeResponse(
            status=302,
            headers={"Location": "https://127.0.0.1/admin"},
        )
        FakePool.responses = [redirect]
        with (
            patch.object(
                helpers.socket,
                "getaddrinfo",
                side_effect=[
                    dns_result("93.184.216.34"),
                    dns_result("93.184.216.34"),
                    dns_result("192.168.1.10"),
                ],
            ),
            patch.object(helpers.urllib3, "HTTPSConnectionPool", FakePool),
            self.assertRaises(helpers.WebsiteIngestionError) as context,
        ):
            helpers.load_website_documents(
                ["https://example.com"],
                deadline=time.monotonic() + 30,
            )
        self.assertEqual(context.exception.category, "blocked_destination")
        self.assertTrue(redirect.closed)
        self.assertTrue(redirect.released)
        self.assertEqual(len(FakePool.instances), 1)

    def test_safe_redirect_is_revalidated_before_following(self):
        redirect = FakeResponse(
            status=302,
            headers={"Location": "https://example.org/final"},
        )
        final = FakeResponse(chunks=(b"<p>Final page content</p>",))
        FakePool.responses = [redirect, final]
        with (
            patch.object(
                helpers.socket,
                "getaddrinfo",
                side_effect=[
                    dns_result("93.184.216.34"),
                    dns_result("93.184.216.34"),
                    dns_result("93.184.216.34"),
                ],
            ),
            patch.object(helpers.urllib3, "HTTPSConnectionPool", FakePool),
        ):
            documents = helpers.load_website_documents(
                ["https://example.com"],
                deadline=time.monotonic() + 30,
            )
        self.assertEqual(len(documents), 1)
        self.assertEqual(len(FakePool.instances), 2)
        self.assertEqual(FakePool.instances[1].host, "93.184.216.34")
        self.assertTrue(redirect.closed)
        self.assertTrue(final.closed)


class WebsiteResponseTests(unittest.TestCase):
    def setUp(self):
        FakePool.instances = []
        FakePool.responses = []

    def _load(self, response, address="93.184.216.34"):
        return self._load_with_report(response, address)[0]

    def _load_with_report(self, response, address="93.184.216.34"):
        FakePool.responses = [response]
        with (
            patch.object(
                helpers.socket,
                "getaddrinfo",
                return_value=dns_result(address),
            ),
            patch.object(helpers.urllib3, "HTTPSConnectionPool", FakePool),
        ):
            return helpers.load_website_documents(
                ["https://example.com"],
                deadline=time.monotonic() + 30,
                return_report=True,
            )

    def test_success_report_uses_existing_extraction_shape(self):
        response = FakeResponse(chunks=(b"<p>Readable page content</p>",))
        documents, report = self._load_with_report(response)
        self.assertEqual(len(documents), 1)
        self.assertEqual(report[0]["filename"], "https://example.com")
        self.assertEqual(report[0]["status"], "loaded")
        self.assertEqual(report[0]["document_count"], 1)
        self.assertGreater(report[0]["extracted_characters"], 0)

    def test_http_failure_has_status_diagnostic(self):
        response = FakeResponse(status=500, headers={"Content-Type": "text/html"})
        with self.assertRaises(helpers.WebsiteIngestionError) as context:
            self._load(response)
        self.assertEqual(context.exception.category, "http_failure")
        self.assertEqual(context.exception.status_code, 500)
        self.assertTrue(response.closed)

    def test_anti_bot_response_is_categorized(self):
        response = FakeResponse(status=403, headers={"Content-Type": "text/html"})
        with self.assertRaises(helpers.WebsiteIngestionError) as context:
            self._load(response)
        self.assertEqual(context.exception.category, "anti_bot")

    def test_empty_page_is_categorized(self):
        response = FakeResponse(chunks=(b"   \n",))
        with self.assertRaises(helpers.WebsiteIngestionError) as context:
            self._load(response)
        self.assertEqual(context.exception.category, "empty_page")
        self.assertTrue(response.closed)

    def test_challenge_body_is_categorized(self):
        response = FakeResponse(
            chunks=(b"<html><body>Just a moment... verify you are human</body></html>",)
        )
        with self.assertRaises(helpers.WebsiteIngestionError) as context:
            self._load(response)
        self.assertEqual(context.exception.category, "anti_bot")


class WebsiteDeadlineTests(unittest.TestCase):
    def setUp(self):
        FakePool.instances = []
        FakePool.responses = []

    def test_expired_deadline_stops_before_dns(self):
        with (
            patch.object(helpers.socket, "getaddrinfo") as resolver,
            self.assertRaises(helpers.WebsiteIngestionError) as context,
        ):
            helpers.load_website_documents(
                ["https://example.com"],
                deadline=time.monotonic() - 1,
            )
        resolver.assert_not_called()
        self.assertEqual(context.exception.category, "timeout")

    def test_deadline_during_stream_closes_response(self):
        clock = [100.0]
        response = FakeResponse(chunks=(b"first", b"second"))

        def stream(*args, **kwargs):
            yield b"first"
            clock[0] = 102.0
            yield b"second"

        response.stream = stream
        FakePool.responses = [response]
        with (
            patch.object(helpers.time, "monotonic", side_effect=lambda: clock[0]),
            patch.object(
                helpers.socket,
                "getaddrinfo",
                return_value=dns_result("93.184.216.34"),
            ),
            patch.object(helpers.urllib3, "HTTPSConnectionPool", FakePool),
            self.assertRaises(helpers.WebsiteIngestionError) as context,
        ):
            helpers.load_website_documents(
                ["https://example.com"],
                deadline=101.0,
            )
        self.assertEqual(context.exception.category, "timeout")
        self.assertTrue(response.closed)
        self.assertTrue(response.released)


class IndexDeadlineTests(unittest.TestCase):
    def test_expired_index_deadline_stops_before_transformations(self):
        with (
            patch.object(llama_index, "run_transformations") as transformations,
            self.assertRaises(TimeoutError),
        ):
            llama_index.create_index(
                [Document(text="content that would otherwise be transformed")],
                deadline=time.monotonic() - 1,
            )
        transformations.assert_not_called()


if __name__ == "__main__":
    unittest.main()
