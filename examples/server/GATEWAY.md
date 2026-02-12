# Gateway Delegation Reference

This document shows reference patterns for deploying `talu serve` behind an auth-capable gateway. The gateway performs authentication/authorization and injects the trusted headers that the server validates.

## Header Contract

The gateway must inject the following headers on every request to protected endpoints:

- `X-Talu-Gateway-Secret`: Shared secret proving traffic came from the gateway.
- `X-Talu-Tenant-Id`: Tenant identifier (maps to `tenants.json`).
- `X-Talu-User-Id`: Optional user identifier for audit logging.

The server expects lowercase header names at runtime (HTTP/2 and `hyper` normalize to lowercase), but case-insensitive matching is supported by HTTP.

## AWS API Gateway (REST + Lambda Authorizer)

### Architecture

1. Client calls API Gateway.
2. Lambda Authorizer validates JWT/IAM and emits a Context map.
3. Integration Request mapping forwards context fields to `X-Talu-*` headers.
4. Integration Request injects a static `X-Talu-Gateway-Secret`.
5. API Gateway forwards to talu server in a private network.

### Lambda Authorizer Output (Pseudo)

```json
{
  "principalId": "user-123",
  "policyDocument": {
    "Version": "2012-10-17",
    "Statement": [{"Action": "execute-api:Invoke", "Effect": "Allow", "Resource": "*"}]
  },
  "context": {
    "taluTenantId": "acme",
    "taluUserId": "user-123"
  }
}
```

### API Gateway Integration Request Mapping

Map authorizer context to headers:

- `X-Talu-Tenant-Id` → `context.authorizer.taluTenantId`
- `X-Talu-User-Id` → `context.authorizer.taluUserId`
- `X-Talu-Gateway-Secret` → static value stored in API Gateway (or from Secrets Manager)

### Notes

- Store the gateway secret in Secrets Manager and inject via stage variables.
- Rotate the secret by updating the gateway and restarting the `talu` server.

## Kubernetes + Cilium (Gateway API or Envoy)

### Architecture

1. External traffic hits a Gateway (Envoy/Istio/Cilium Gateway API).
2. AuthN/AuthZ enforced via JWT or external auth filter.
3. Gateway injects `X-Talu-*` headers.
4. Cilium NetworkPolicy restricts `talu` pods to gateway-only ingress.

### Cilium NetworkPolicy (L3/L4 Restriction)

```yaml
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: talu-ingress-gateway-only
spec:
  endpointSelector:
    matchLabels:
      app: talu
  ingress:
    - fromEndpoints:
        - matchLabels:
            app: api-gateway
      toPorts:
        - ports:
            - port: "8258"
              protocol: TCP
```

### Gateway Header Injection (Gateway API / Envoy)

Example Gateway API filter snippet:

```yaml
filters:
  - type: RequestHeaderModifier
    requestHeaderModifier:
      add:
        - name: X-Talu-Gateway-Secret
          value: "${TALU_GATEWAY_SECRET}"
        - name: X-Talu-Tenant-Id
          value: "{jwt.claims.tenant_id}"
        - name: X-Talu-User-Id
          value: "{jwt.claims.sub}"
```

### Notes

- Pair header injection with JWT validation to prevent client spoofing.
- Rotate the gateway secret by updating the gateway deployment and server config.

## Local Development

Start the server with auth enabled:

```bash
./zig-out/bin/talu serve \
  --gateway-secret "dev-secret" \
  --tenant-config ./tenants.json.example
```

Then send requests with headers:

```bash
curl -H 'X-Talu-Gateway-Secret: dev-secret' \
  -H 'X-Talu-Tenant-Id: acme' \
  http://127.0.0.1:8258/v1/models
```
