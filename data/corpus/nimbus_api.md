# Nimbus Analytics — Developer API

## Authentication

The Nimbus REST API authenticates with a bearer token. Generate an API key under
Settings → Developer. Pass it in the `Authorization: Bearer <key>` header on every
request. Keys can be scoped to read-only or read-write.

## Base URL and versioning

The API base URL is `https://api.nimbus.example/v1`. The version is pinned in the
URL path; breaking changes ship under a new version prefix and the previous version
is supported for 12 months after a new one is released.

## Rate limits

API requests are limited to 100 requests per minute on Starter, 600 per minute on
Growth, and 2,000 per minute on Enterprise. Exceeding the limit returns HTTP 429 with
a `Retry-After` header indicating how many seconds to wait.

## Pagination

List endpoints return at most 50 items per page. Use the `cursor` query parameter
returned in the `next_cursor` field to fetch subsequent pages. There is no offset-based
pagination.

## Webhooks

Webhooks can be registered for the `report.completed` and `refresh.failed` events.
Each webhook delivery is signed with an HMAC-SHA256 signature in the
`X-Nimbus-Signature` header, which the receiver should verify.
